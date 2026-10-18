"""
compound_pilot.py  --  does a compound-token model beat the triplet model at EQUAL WALL-CLOCK?

One process = one GPU = one model. Run the variants side by side (two T4s -> two runs at once):

  CUDA_VISIBLE_DEVICES=0 python compound_pilot.py --model triplet  --notes 256 --run_name pilot-triplet-256  ... &
  CUDA_VISIBLE_DEVICES=1 python compound_pilot.py --model compound --notes 256 --run_name pilot-compound-256 ... &

Models (identical small backbone: 6 layers, 384 dim, same RoPE with real time / pitch / index groups)
  triplet  : the model from music_common.py, 3 positions per note (shift, pitch, duration)
  compound : ONE position per note. Position i sees note i-1 (sum of shift+pitch+duration embeddings).
             It predicts note i as shift | h  ->  pitch | h, shift  ->  duration | h, shift, pitch,
             with small MLP sub-heads (teacher-forced). Same discrete distribution as the triplet model,
             so bits/note is directly comparable.

Fair comparison: the same wall-clock budget, the same ARIA corpus and augmentation, the same MAESTRO validation
and test windows, and the same protocol (only notes in the second half of each window are scored, so each scored
note has at least half a window of context). The compound model is ~3x cheaper per note, so in the same time it
sees ~3x more notes -- that is exactly the advantage being tested.
Metric to compare: bits/note on MAESTRO val/test (and pitch PPL, the hard part).
"""
import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.amp import GradScaler, autocast

from music_common import (VOCAB, Block, Logger, MusicRoPE, MusicTransformer, TokenEmbedding, TypeStats,
                          WindowDataset, build_model, concat_songs, count_params, evaluate, flatten_summary,
                          format_summary, load_corpus, load_or_build_maestro_cache, lr_at, make_optimizer,
                          masked_ce)
from pretrain_aria import fetch_corpus


# ----------------------------------------------------------------------------------------------------------
class CompoundTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, num_layers=6, dropout=0.0, rope_dims=(32, 16, 16)):
        super().__init__()
        D = embed_dim
        self.rope = MusicRoPE(D // num_heads, rope_dims)
        self.shift_emb = nn.Embedding(128, D)
        self.dur_emb = nn.Embedding(128, D)
        self.pitch = TokenEmbedding(128, D, factored=True)          # .weight() -> (128, D) factored pitch matrix
        self.bos = nn.Parameter(torch.zeros(D))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(D, num_heads, dropout, self.rope.active_dims) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(D)
        self.sub_p = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))
        self.sub_d = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, D))
        self.apply(MusicTransformer._init)
        nn.init.normal_(self.pitch.pitch_class.weight, 0.0, 0.01)
        nn.init.normal_(self.pitch.octave.weight, 0.0, 0.01)
        nn.init.normal_(self.bos, 0.0, 0.02)
        res_std = 0.02 / math.sqrt(2 * num_layers)
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, 0.0, res_std)
            nn.init.normal_(b.mlp[2].weight, 0.0, res_std)

    def forward(self, x, active):
        """x: (B, 3N) tokens [shift, pitch, dur] * N (window starts on a shift token); active: (B, 3N).
        Returns logits for shift / pitch / duration of every note plus the target classes."""
        B, S = x.shape
        sh = x[:, 0::3] - 128                                       # (B,N) classes 0..127
        pi = x[:, 1::3]
        du = x[:, 2::3] - 256
        Wp = self.pitch.weight()
        e_s = F.embedding(sh, self.shift_emb.weight)
        e_p = F.embedding(pi, Wp)
        e_d = F.embedding(du, self.dur_emb.weight)
        e_note = e_s + e_p + e_d
        h = torch.cat([self.bos.expand(B, 1, -1), e_note[:, :-1]], dim=1)          # position i <- note i-1
        ticks = F.pad(sh.cumsum(1)[:, :-1], (1, 0))                                 # onset time of note i-1
        pitch_pos = torch.cat([active[:, :1], pi[:, :-1]], dim=1)                   # pitch of note i-1
        cs = self.rope.build_positions(ticks, pitch_pos)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h, cs)
        h = self.ln_f(h)
        lg_s = F.linear(h, self.shift_emb.weight)
        lg_p = F.linear(h + self.sub_p(torch.cat([h, e_s], dim=-1)), Wp)
        lg_d = F.linear(h + self.sub_d(torch.cat([h, e_s, e_p], dim=-1)), self.dur_emb.weight)
        return lg_s, lg_p, lg_d, (sh, pi, du)


def per_note_nll(lg_s, lg_p, lg_d, sh, pi, du):
    f = lambda lg, t: F.cross_entropy(lg.float().transpose(1, 2), t, reduction="none")      # (B,N)
    return f(lg_s, sh), f(lg_p, pi), f(lg_d, du)


def update_compound_stats(st, lg_s, lg_p, lg_d, sh, pi, du, ns, npi, nd, k=0):
    """Re-uses TypeStats by mapping each class back to token-id space (so the type is inferred correctly)."""
    st.update(lg_s.argmax(-1)[:, k:] + 128, sh[:, k:] + 128, ns[:, k:])
    st.update(lg_p.argmax(-1)[:, k:], pi[:, k:], npi[:, k:])
    st.update(lg_d.argmax(-1)[:, k:] + 256, du[:, k:] + 256, nd[:, k:])


@torch.no_grad()
def evaluate_compound(model, loader, device, score_from_notes):
    model.eval()
    st = TypeStats(device)
    for x, y, a in loader:
        x, a = x.to(device, non_blocking=True), a.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16):
            lg_s, lg_p, lg_d, (sh, pi, du) = model(x, a)
        ns, npi, nd = per_note_nll(lg_s, lg_p, lg_d, sh, pi, du)
        update_compound_stats(st, lg_s, lg_p, lg_d, sh, pi, du, ns, npi, nd, score_from_notes)
    return st.summary()


# ----------------------------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["triplet", "compound"], required=True)
    p.add_argument("--notes", type=int, default=256, help="notes per window (even). triplet window = 3*notes tokens")
    p.add_argument("--corpus_dir", default="")
    p.add_argument("--corpus_artifact", default="")
    p.add_argument("--maestro_dir", default="data")
    p.add_argument("--out_dir", default="pilot_out")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--transpose", type=int, default=5)
    p.add_argument("--time_stretch", action="store_true")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--rope_dims", type=str, default="32,16,16")
    p.add_argument("--val_windows", type=int, default=1536, help="subset used for the periodic evals")
    p.add_argument("--eval_every_min", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--run_name", default="")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="music-rnn-ddp")
    p.add_argument("--wandb_entity", default="riya-rajbhut-student")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.notes % 2 == 0, "--notes must be even"
    fetch_corpus(args)
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.run_name or f"pilot-{args.model}-{args.notes}"

    seq = 3 * args.notes
    tokens, offsets, keep = load_corpus(args.corpus_dir)
    lens = np.diff(offsets)
    train_ids = np.where(keep & (lens >= seq + 3))[0]
    train_ds = WindowDataset(tokens, offsets, train_ids, seq, seq, jitter=True, transpose=args.transpose,
                             time_stretch=args.time_stretch)
    songs = load_or_build_maestro_cache(args.maestro_dir, workers=os.cpu_count() or 4)
    ev = {}
    for split in ("validation", "test"):
        tok, off = concat_songs(songs[split])
        ev[split] = (tok, off, np.arange(len(off) - 1))
    half = seq // 2
    val_small = WindowDataset(*ev["validation"], seq, half, max_windows=args.val_windows)
    val_full = WindowDataset(*ev["validation"], seq, half)
    test_full = WindowDataset(*ev["test"], seq, half)
    print(f"train windows {len(train_ds)} ({train_ds.n_tokens() / 1e6:.0f}M tokens/pass) | "
          f"val {len(val_full)} | test {len(test_full)} | window = {args.notes} notes ({seq} tokens)", flush=True)

    rope_dims = [int(v) for v in args.rope_dims.split(",")]
    if args.model == "triplet":
        cfg = dict(num_tokens=VOCAB, embed_dim=args.embed_dim, num_heads=args.num_heads,
                   num_layers=args.num_layers, factored=True, rope_dims=rope_dims)
        model = build_model(cfg, dropout=args.dropout).to(device)
    else:
        model = CompoundTransformer(args.embed_dim, args.num_heads, args.num_layers, args.dropout, rope_dims).to(device)
    print(f"{args.model}: {count_params(model) / 1e6:.1f}M params", flush=True)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    scaler = GradScaler("cuda")
    logger = Logger(not args.no_wandb, project=args.wandb_project, entity=args.wandb_entity, name=name,
                    config=vars(args))

    def loader_for(ds):
        return data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    def evaluate_any(ds):
        if args.model == "triplet":
            return evaluate(model, loader_for(ds), device, score_from=half)
        return evaluate_compound(model, loader_for(ds), device, score_from_notes=half // 3)

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   num_workers=args.num_workers, pin_memory=True,
                                   persistent_workers=args.num_workers > 0,
                                   prefetch_factor=4 if args.num_workers > 0 else None)
    budget = args.hours * 3600.0
    tr = TypeStats(device)
    step, t0, t_log, next_eval, first = 0, time.time(), time.time(), args.eval_every_min * 60.0, True
    done = False
    model.train()
    while not done:
        for x, y, a in train_loader:
            elapsed = time.time() - t0
            if elapsed >= budget:
                done = True
                break
            lr = lr_at(args.lr, step, args.warmup_steps, elapsed / budget, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, a = x.to(device, non_blocking=True), y.to(device, non_blocking=True), a.to(device, non_blocking=True)
            if args.model == "triplet":
                with autocast("cuda", dtype=torch.float16):
                    logits, _ = model(x, a)
                loss, nll, pred = masked_ce(logits, y, x, 0.0)
                tr.update(pred, y, nll)
            else:
                with autocast("cuda", dtype=torch.float16):
                    lg_s, lg_p, lg_d, (sh, pi, du) = model(x, a)
                ns, npi, nd = per_note_nll(lg_s, lg_p, lg_d, sh, pi, du)
                loss = (ns + npi + nd).mean() / 3.0                  # same scale as the per-token triplet loss
                update_compound_stats(tr, lg_s.detach(), lg_p.detach(), lg_d.detach(), sh, pi, du,
                                      ns.detach(), npi.detach(), nd.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if first:
                assert float(loss) == float(loss), "NaN loss on the first step"
                first = False

            if step % args.log_every == 0:
                s = tr.summary()
                if not math.isfinite(s["nll"]):
                    raise RuntimeError("non-finite training loss")
                nps = args.log_every * args.batch_size * args.notes / (time.time() - t_log)
                print(f"[step {step}] {format_summary(s)} | lr {lr:.2e} | {nps / 1e3:.0f}k notes/s | "
                      f"{elapsed / 60:.1f}/{budget / 60:.0f} min", flush=True)
                d = flatten_summary("train", s)
                d.update({"lr": lr, "notes_per_sec": nps, "notes_seen": step * args.batch_size * args.notes})
                logger.log(d, step=step)
                tr.reset()
                t_log = time.time()
            if elapsed >= next_eval:
                res = evaluate_any(val_small)
                model.train()
                print(f"[step {step}] MAESTRO-val (subset) {format_summary(res)}", flush=True)
                logger.log(flatten_summary("val_subset", res), step=step)
                next_eval += args.eval_every_min * 60.0
                t_log = time.time()

    val = evaluate_any(val_full)
    test = evaluate_any(test_full)
    notes_seen = step * args.batch_size * args.notes
    print(f"\n=== FINAL {name} | {step} steps | {notes_seen / 1e6:.0f}M notes seen ===\n"
          f"VAL  {format_summary(val)}\nTEST {format_summary(test)}", flush=True)
    logger.log({**flatten_summary("val", val), **flatten_summary("test", test)}, step=step)
    logger.summary({"final/val_bits_per_note": val["bits_per_note"], "final/test_bits_per_note": test["bits_per_note"],
                    "final/notes_seen": notes_seen})
    json.dump({"name": name, "model": args.model, "notes": args.notes, "steps": step, "notes_seen": notes_seen,
               "val": val, "test": test}, open(out_dir / f"{name}.json", "w"), indent=2)
    logger.finish()


if __name__ == "__main__":
    main()
