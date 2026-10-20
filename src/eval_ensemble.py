"""
eval_ensemble.py  --  inference-only pitch evaluation on MAESTRO with three optional boosters:

  1. transposition ensemble : average pitch distributions over inputs shifted by k semitones (mapped back)
  2. checkpoint ensemble    : average over several fine-tuned checkpoints (pass several --ckpt)
  3. interval cache         : mix in the pitch proposed by the most recent earlier occurrence of the last n
                              pitch intervals in the SAME piece (causal; mixing weight tuned on validation)

Protocol: windows of 768 tokens, hop 384, only notes in the second half of each window are scored
(same as `--score_from 384 --eval_hop 384` in finetune_maestro.py). The cache looks at the whole earlier piece,
so report the cache rows as a separate, clearly labelled protocol.

Usage (run from the folder that holds music_common.py):
  python eval_ensemble.py --canon --ckpt artifacts/best_maestro_canon.pt
  python eval_ensemble.py --canon --ckpt artifacts/best_maestro_canon.pt artifacts/best_maestro_canon_v3.pt
  python eval_ensemble.py --ckpt artifacts/best_maestro_aria_init.pt            # old (micro-timing) token order

IMPORTANT: pass --canon for checkpoints trained with CANON_CHORDS=1, and never for the others.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--canon", action="store_true", help="use the canonical chord order tokenisation (CANON_CHORDS=1)")
ap.add_argument("--ks", default="-3,-2,-1,0,1,2,3", help="transposition shifts (must include 0)")
ap.add_argument("--cache_n", type=int, default=6, help="number of pitch intervals in the cache context")
ap.add_argument("--cache_dist", type=int, default=128, help="only use earlier matches at least this many notes back")
ap.add_argument("--maestro_dir", default="data")
args = ap.parse_args()
if args.canon:
    os.environ["CANON_CHORDS"] = "1"          # must be set before the MAESTRO cache is loaded

import numpy as np
import torch
from torch.amp import autocast

from music_common import (WindowDataset, build_model, concat_songs, load_or_build_maestro_cache, load_weights)

dev = torch.device("cuda:0")
KS = tuple(int(k) for k in args.ks.split(","))
assert 0 in KS, "--ks must contain 0"
LO, HI = 16, 113                              # a shifted view is skipped if a pitch would leave this range
SEQ, HOP, SCORE_FROM = 768, 384, 384
WS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]  # candidate cache weights

models = []
for p in args.ckpt:
    ck = torch.load(p, map_location="cpu", weights_only=True)
    m = build_model(ck["model_cfg"], dropout=0.0).to(dev)
    load_weights(m, ck["model_state_dict"])
    models.append(m.eval())
    print(f"loaded {p} | meta {ck.get('meta')}", flush=True)
songs = load_or_build_maestro_cache(args.maestro_dir, workers=4)
print(f"tokenisation: {'canonical chord order' if args.canon else 'original order'}", flush=True)


def view_logp(model, x, a, k):
    """pitch log-probs for the input transposed by k, mapped back to the original key: (B,S,128)"""
    xs = torch.where(x < 128, x + k, x)
    with autocast("cuda", dtype=torch.float16):
        logits, _ = model(xs, a + k)
    idx = torch.arange(128, device=x.device) + k
    ok = (idx >= 0) & (idx < 128)
    lg = logits[..., :128].float()[..., idx.clamp(0, 127)].masked_fill(~ok, float("-inf"))
    return torch.log_softmax(lg, -1)


def window_valid(x, y, k):
    if k == 0:
        return torch.ones(len(x), dtype=torch.bool, device=x.device)
    allt = torch.cat([x, y[:, -1:]], 1)
    p = torch.where(allt < 128, allt, torch.full_like(allt, 64))
    return ((p + k).amin(1) >= LO) & ((p + k).amax(1) <= HI)


def interval_guess(p, n, min_dist):
    """for each note i: pitch proposed by the most recent earlier occurrence of the last n intervals (-1 = none)"""
    g = np.full(len(p), -1, np.int64)
    d = np.diff(p)
    seen = {}
    for i in range(n + 1, len(p)):
        key = d[i - n - 1:i - 1].tobytes()
        e = seen.get(key)
        if e is not None and i - e[0] > min_dist:
            c = int(p[i - 1]) + e[1]
            if 0 <= c < 128:
                g[i] = c
        seen[key] = (i, int(p[i]) - int(p[i - 1]))
    return g


def collect(split):
    P_single, P_ens, Ts, Gs = [], [], [], []
    k0 = KS.index(0)
    for t in songs[split]:
        t = np.asarray(t)
        pit = t[1::3].astype(np.int16)
        pit_t = torch.as_tensor(pit.astype(np.int64), device=dev)
        g_all = torch.as_tensor(interval_guess(pit, args.cache_n, args.cache_dist), device=dev)
        tok1, off1 = concat_songs([t])
        ds = WindowDataset(tok1, off1, np.arange(1), SEQ, HOP)
        for j0 in range(0, len(ds), 16):
            js = list(range(j0, min(j0 + 16, len(ds))))
            items = [ds[j] for j in js]
            for j, (x, _, _) in zip(js, items):
                assert np.array_equal(torch.as_tensor(x).numpy(), t[j * HOP:j * HOP + SEQ]), \
                    "window start assumption broken"
            X, Y, A = (torch.stack([torch.as_tensor(it[q]) for it in items]).to(dev) for q in range(3))
            st = torch.tensor([j * HOP for j in js], device=dev)
            with torch.no_grad():
                val = torch.stack([window_valid(X, Y, k) for k in KS]).float()                    # (K,B)
                views = []
                for m in models:
                    views.append(torch.stack([view_logp(m, X, A, k) for k in KS])[:, :, SCORE_FROM:].exp())
                single = views[0][k0]                                                              # first model, k=0
                V = torch.cat(views, 0)                                                            # (M*K,B,S',128)
                W = val.repeat(len(models), 1)                                                     # (M*K,B)
                ens = (V * W[:, :, None, None]).sum(0) / W.sum(0)[:, None, None]
            yy = Y[:, SCORE_FROM:]
            pm = yy < 128
            tt = torch.arange(SCORE_FROM, SEQ, device=dev)[None].expand(len(js), -1)
            m_idx = ((st[:, None] + tt) // 3)[pm]
            tgt = yy[pm]
            assert (pit_t[m_idx] == tgt).all(), "note-index mapping is wrong"
            P_single.append(single[pm].half().cpu())
            P_ens.append(ens[pm].half().cpu())
            Ts.append(tgt.cpu())
            Gs.append(g_all[m_idx].cpu())
    return torch.cat(P_single), torch.cat(P_ens), torch.cat(Ts), torch.cat(Gs)


def score(P, T, G, w):
    ok = 0.0
    nll = 0.0
    for lo in range(0, len(T), 200_000):
        p = P[lo:lo + 200_000].to(dev).float()
        t = T[lo:lo + 200_000].to(dev)
        g = G[lo:lo + 200_000].to(dev)
        rows = (g >= 0).nonzero().squeeze(1)
        if w > 0 and len(rows):
            p[rows] *= (1 - w)
            p[rows, g[rows]] += w
        nll += float(-torch.log(p.gather(1, t[:, None]).squeeze(1).clamp_min(1e-12)).sum())
        ok += float((p.argmax(1) == t).sum())
    return ok / len(T) * 100, float(np.exp(nll / len(T)))


data = {s: collect(s) for s in ("validation", "test")}
print(f"\nmodels: {len(models)} | transposition shifts: {KS} | cache: n={args.cache_n}, dist>{args.cache_dist}")
res = {}
for s, (Ps, Pe, T, G) in data.items():
    res[s] = {"single model (k=0)": score(Ps, T, G, 0.0), "ensemble": score(Pe, T, G, 0.0)}
    print(f"{s}: scored notes {len(T)} | cache covers {(G >= 0).float().mean() * 100:.1f}%")
Pv, Tv, Gv = data["validation"][1:]
grid = [(score(Pv, Tv, Gv, w), w) for w in WS]
(_, _), w_best = min(grid, key=lambda r: r[0][1])           # best weight = lowest validation PPL
for s, (Ps, Pe, T, G) in data.items():
    res[s][f"ensemble + cache (w={w_best})"] = score(Pe, T, G, w_best)
print(f"\ncache weight chosen on validation: {w_best}\n")
print(f"{'':36s}{'VAL acc':>10s}{'VAL ppl':>10s}{'TEST acc':>11s}{'TEST ppl':>10s}")
for name in res["validation"]:
    v, t = res["validation"][name], res["test"][name]
    print(f"{name:36s}{v[0]:9.2f}%{v[1]:10.3f}{t[0]:10.2f}%{t[1]:10.3f}")
print("\nnote: pitch only; the cache row uses the whole earlier piece, so report it as a separate protocol.")