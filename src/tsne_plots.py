"""
tsne_plots.py  --  t-SNE views of what the trained model has learned.

Two figures (choose with --what):

 1) embeddings : t-SNE of the learned token vectors (no data needed)
      - the 88 piano pitches (MIDI 21-108), coloured by pitch class and by octave
      - the 128 time-shift tokens and the 128 duration tokens, coloured by their value
 3) instances  : t-SNE of the model's vector at each occurrence ("instance") of a pitch token in real music
      (about --per_pitch instances for each of the 88 pitches), coloured by the pitch's chroma and octave.
      Use --layer 0, 3, 6, 11 to see how token identity is kept or lost with depth.
 2) hidden     : t-SNE of the model's internal vector at positions where the next token is a PITCH
      (the shift-token positions), taken from MAESTRO validation windows (second half of each window), coloured by
      the true next pitch class, the true next pitch, whether the prediction is correct, and the confidence.

Usage (from the folder that holds music_common.py):
  python tsne_plots.py --canon --ckpt /kaggle/working/artifacts/best_maestro_canon.pt --out_dir tsne_out
  python tsne_plots.py --canon --ckpt ... --what hidden --layer 6        # an intermediate block (0-11); -1 = final
  python tsne_plots.py --ckpt /kaggle/working/artifacts/best_maestro_aria_init.pt     # original-order model

Use --canon only for checkpoints trained with CANON_CHORDS=1.
t-SNE shows neighbourhoods, not distances: do not read meaning into the size of gaps or the position of clusters.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--canon", action="store_true")
ap.add_argument("--what", choices=["embeddings", "hidden", "instances", "both", "all"], default="both",
                help="both = embeddings + hidden; all = embeddings + hidden + instances")
ap.add_argument("--per_pitch", type=int, default=40, help="instances mode: contextual vectors sampled per pitch")
ap.add_argument("--layer", type=int, default=-1, help="-1 = after the final layer norm, 0..11 = output of that block")
ap.add_argument("--n_points", type=int, default=6000, help="hidden-state points to embed (about this many)")
ap.add_argument("--perplexity", type=float, default=30.0, help="for the hidden-state t-SNE (embeddings use 15-20)")
ap.add_argument("--out_dir", default="tsne_out")
ap.add_argument("--maestro_dir", default="data")
ap.add_argument("--seed", type=int, default=53)
args = ap.parse_args()
if args.canon:
    os.environ["CANON_CHORDS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.amp import autocast

from music_common import WindowDataset, build_model, concat_songs, load_or_build_maestro_cache, load_weights

os.makedirs(args.out_dir, exist_ok=True)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_COLORS = plt.cm.hsv(np.arange(12) / 12.0)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
model = build_model(ck["model_cfg"], dropout=0.0).to(dev)
load_weights(model, ck["model_state_dict"])
model.eval()
print(f"loaded {args.ckpt} | meta {ck.get('meta')} | tokenisation: "
      f"{'canonical chord order' if args.canon else 'original order'}", flush=True)


def run_tsne(X, perplexity):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] > 50 and X.shape[0] > 50:
        X = PCA(n_components=50, random_state=args.seed).fit_transform(X)
    kw = dict(n_components=2, perplexity=min(perplexity, (len(X) - 1) / 3.0), init="pca",
              learning_rate="auto", random_state=args.seed)
    try:
        ts = TSNE(max_iter=1000, **kw)
    except TypeError:                                   # older scikit-learn
        ts = TSNE(n_iter=1000, **kw)
    return ts.fit_transform(X)


def pc_legend(ax):
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PC_COLORS[i], label=NAMES[i]) for i in range(12)],
              ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.17), frameon=False)


def embeddings_figure():
    with torch.no_grad():
        W = model.embed.weight().float().cpu().numpy()          # rows 0-127 pitch, 128-255 shift, 256-383 duration
    pitch = np.arange(21, 109)
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))

    xy = run_tsne(W[pitch], 15)
    pc = pitch % 12
    ax[0, 0].scatter(xy[:, 0], xy[:, 1], c=PC_COLORS[pc], s=60)
    for (x, y), p in zip(xy, pitch):
        ax[0, 0].annotate(f"{NAMES[p % 12]}{p // 12 - 1}", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[0, 0].set_title("Pitch vectors (88 piano keys), coloured by pitch class")
    pc_legend(ax[0, 0])
    sc = ax[0, 1].scatter(xy[:, 0], xy[:, 1], c=pitch // 12 - 1, cmap="viridis", s=60)
    ax[0, 1].set_title("Same layout, coloured by octave")
    fig.colorbar(sc, ax=ax[0, 1], label="octave (C4 = middle C octave)")

    for k, (name, rows, cmap) in enumerate([("Time-shift tokens (0-127 ticks)", W[128:256], "plasma"),
                                            ("Duration tokens (0-127 ticks)", W[256:384], "plasma")]):
        a = ax[1, k]
        xy2 = run_tsne(rows, 20)
        sc = a.scatter(xy2[:, 0], xy2[:, 1], c=np.arange(128), cmap=cmap, s=40)
        for v in range(0, 128, 8):
            a.annotate(str(v), xy2[v], fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_title(name)
        fig.colorbar(sc, ax=a, label="value in ticks (1 tick = 31.25 ms)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("t-SNE of the learned token vectors", fontsize=15)
    fig.tight_layout()
    path = os.path.join(args.out_dir, "tsne_embeddings.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}", flush=True)


def hidden_figure():
    cap = {}
    target = model.ln_f if args.layer < 0 else model.blocks[args.layer]
    hook = target.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    songs = load_or_build_maestro_cache(args.maestro_dir, workers=4)
    tok, off = concat_songs(songs["validation"])
    ds = WindowDataset(tok, off, np.arange(len(off) - 1), 768, 384)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(len(ds), max(args.n_points // 2, 1)), replace=False)
    loader = data.DataLoader(data.Subset(ds, idx.tolist()), batch_size=32, num_workers=2)
    N0 = 128                                                    # first scored note (second half of the window)
    H, T, PR, CF = [], [], [], []
    with torch.no_grad():
        for x, y, a in loader:
            x, y, a = x.to(dev), y.to(dev), a.to(dev)
            with autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, _ = model(x, a)
            hid = cap["h"][:, 0::3][:, N0:].float()             # vectors at shift-token positions (next token = pitch)
            prob = torch.softmax(logits[:, 0::3, :128].float()[:, N0:], -1)
            tgt = y[:, 0::3][:, N0:]
            B, N, D = hid.shape
            pick = torch.stack([torch.randperm(N, device=dev)[:2] for _ in range(B)])
            gi = torch.arange(B, device=dev)[:, None]
            H.append(hid[gi, pick].reshape(-1, D).cpu())
            T.append(tgt[gi, pick].reshape(-1).cpu())
            PR.append(prob.argmax(-1)[gi, pick].reshape(-1).cpu())
            CF.append(prob.max(-1).values[gi, pick].reshape(-1).cpu())
    hook.remove()
    H, T, PR, CF = (torch.cat(v).numpy() for v in (H, T, PR, CF))
    print(f"{len(H)} hidden vectors | layer {'final' if args.layer < 0 else args.layer} | "
          f"accuracy of these points {np.mean(PR == T) * 100:.1f}%", flush=True)
    xy = run_tsne(H, args.perplexity)

    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    ax[0, 0].scatter(xy[:, 0], xy[:, 1], c=PC_COLORS[T % 12], s=6, alpha=0.7)
    ax[0, 0].set_title("Coloured by the TRUE next pitch class")
    pc_legend(ax[0, 0])
    sc = ax[0, 1].scatter(xy[:, 0], xy[:, 1], c=T, cmap="viridis", s=6, alpha=0.7)
    ax[0, 1].set_title("Coloured by the TRUE next pitch (register)")
    fig.colorbar(sc, ax=ax[0, 1], label="MIDI pitch")
    ok = PR == T
    ax[1, 0].scatter(xy[ok, 0], xy[ok, 1], c="tab:blue", s=6, alpha=0.5, label=f"correct ({ok.mean() * 100:.0f}%)")
    ax[1, 0].scatter(xy[~ok, 0], xy[~ok, 1], c="tab:red", s=6, alpha=0.5, label=f"wrong ({(~ok).mean() * 100:.0f}%)")
    ax[1, 0].legend(markerscale=3)
    ax[1, 0].set_title("Prediction correct or wrong")
    sc = ax[1, 1].scatter(xy[:, 0], xy[:, 1], c=CF, cmap="magma", s=6, alpha=0.7, vmin=0, vmax=1)
    ax[1, 1].set_title("Model confidence (top predicted probability)")
    fig.colorbar(sc, ax=ax[1, 1], label="probability")
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"t-SNE of hidden vectors where the next token is a pitch "
                 f"({'final layer' if args.layer < 0 else f'block {args.layer}'})", fontsize=15)
    fig.tight_layout()
    path = os.path.join(args.out_dir, f"tsne_hidden_{'final' if args.layer < 0 else f'block{args.layer}'}.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}", flush=True)


def instances_figure():
    cap = {}
    target = model.ln_f if args.layer < 0 else model.blocks[args.layer]
    hook = target.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    songs = load_or_build_maestro_cache(args.maestro_dir, workers=4)
    tok, off = concat_songs(songs["validation"])
    ds = WindowDataset(tok, off, np.arange(len(off) - 1), 768, 384)
    rng = np.random.default_rng(args.seed)
    loader = data.DataLoader(data.Subset(ds, rng.permutation(len(ds)).tolist()), batch_size=32, num_workers=2)
    N0 = 128
    pitches = np.arange(21, 109)
    count = {int(p): 0 for p in pitches}
    H, L = [], []
    with torch.no_grad():
        for x, y, a in loader:
            x, a = x.to(dev), a.to(dev)
            with autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                model(x, a)
            hid = cap["h"][:, 1::3][:, N0:].float().cpu().numpy()      # vectors at pitch-token positions
            lab = x[:, 1::3][:, N0:].cpu().numpy()
            hid, lab = hid.reshape(-1, hid.shape[-1]), lab.reshape(-1)
            for p in pitches:
                need = args.per_pitch - count[int(p)]
                if need <= 0:
                    continue
                idx = np.where(lab == p)[0][:need]
                if len(idx):
                    H.append(hid[idx]); L.append(lab[idx]); count[int(p)] += len(idx)
            if all(c >= args.per_pitch for c in count.values()):
                break
    hook.remove()
    H, L = np.concatenate(H), np.concatenate(L)
    short = [p for p, c in count.items() if c < args.per_pitch]
    print(f"{len(H)} instances of {len(np.unique(L))} pitches | layer {'final' if args.layer < 0 else args.layer} | "
          f"{len(short)} rare pitches have fewer than {args.per_pitch} instances", flush=True)
    xy = run_tsne(H, args.perplexity)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    ax[0].scatter(xy[:, 0], xy[:, 1], c=PC_COLORS[L % 12], s=8, alpha=0.8)
    ax[0].set_title("Token instances, coloured by chroma (pitch class)")
    pc_legend(ax[0])
    sc = ax[1].scatter(xy[:, 0], xy[:, 1], c=L // 12 - 1, cmap="viridis", s=8, alpha=0.8)
    ax[1].set_title("Token instances, coloured by octave")
    fig.colorbar(sc, ax=ax[1], label="octave")
    for p in np.unique(L):
        c = np.median(xy[L == p], axis=0)
        ax[0].annotate(f"{NAMES[p % 12]}{p // 12 - 1}", c, fontsize=6, ha="center", va="center")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"t-SNE of the model's vector at each pitch-token position "
                 f"({'final layer' if args.layer < 0 else f'block {args.layer}'})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(args.out_dir, f"tsne_instances_{'final' if args.layer < 0 else f'block{args.layer}'}.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}", flush=True)


if args.what in ("embeddings", "both", "all"):
    embeddings_figure()
if args.what in ("hidden", "both", "all"):
    hidden_figure()
if args.what in ("instances", "all"):
    instances_figure()