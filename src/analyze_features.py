"""
analyze_features.py  --  put numbers on what the t-SNE picture seems to show. NumPy only (no torch, no GPU).

Input: hidden_feats_<layer>.npz written by tsne_plots.py (hidden vectors where the next token is a pitch, with
the true next pitch T, the predicted pitch PR and the model's confidence CF).

Prints:
  1) accuracy, error share and mean confidence per register bin (is the error mass really around MIDI ~50?)
  2) calibration: accuracy per confidence bin
  3) nearest-neighbour agreement in the ORIGINAL 768-d space (not the 2-D picture): of each point's k nearest
     neighbours (cosine), how many share its pitch class / octave / exact pitch / correctness, against chance.

Usage:  python analyze_features.py --features /kaggle/working/tsne_out/hidden_feats_final.npz
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--k", type=int, default=10)
args = ap.parse_args()

z = np.load(args.features)
H, T, PR, CF = z["H"].astype(np.float32), z["T"], z["PR"], z["CF"]
n = len(T)
ok = PR == T
print(f"{n} points | accuracy {ok.mean() * 100:.1f}% | mean confidence {CF.mean():.3f}\n")

print("1) by register of the TRUE next pitch")
edges = [20, 36, 48, 60, 72, 84, 109]
print(f"{'MIDI range':>12s}{'share of notes':>16s}{'accuracy':>10s}{'share of errors':>17s}{'mean conf':>11s}")
n_err = (~ok).sum()
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (T >= lo) & (T < hi)
    if m.sum() == 0:
        continue
    print(f"{f'{lo}-{hi - 1}':>12s}{m.mean() * 100:15.1f}%{ok[m].mean() * 100:9.1f}%"
          f"{(~ok[m]).sum() / n_err * 100:16.1f}%{CF[m].mean():11.3f}")
print("   (if 'share of errors' is close to 'share of notes', errors simply follow where the notes are)\n")

print("2) calibration")
cedges = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
print(f"{'confidence':>12s}{'share':>9s}{'accuracy':>10s}")
for lo, hi in zip(cedges[:-1], cedges[1:]):
    m = (CF >= lo) & (CF < hi)
    if m.sum():
        print(f"{f'{lo:.2f}-{min(hi, 1):.2f}':>12s}{m.mean() * 100:8.1f}%{ok[m].mean() * 100:9.1f}%")

print(f"\n3) nearest-neighbour agreement in the original space (k = {args.k}, cosine)")
Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
S = Hn @ Hn.T
np.fill_diagonal(S, -np.inf)
nn = np.argpartition(-S, args.k, axis=1)[:, :args.k]
labels = {"pitch class": T % 12, "octave": T // 12, "exact pitch": T, "prediction correct": ok.astype(int),
          "confidence > 0.8": (CF > 0.8).astype(int)}
print(f"{'label':>20s}{'neighbours that agree':>24s}{'chance':>9s}")
for name, lab in labels.items():
    agree = (lab[nn] == lab[:, None]).mean()
    _, cnt = np.unique(lab, return_counts=True)
    chance = ((cnt / n) ** 2).sum()
    print(f"{name:>20s}{agree * 100:23.1f}%{chance * 100:8.1f}%")
w = ~ok
print(f"\nAmong WRONG points, {(~ok[nn[w]]).mean() * 100:.1f}% of their neighbours are also wrong "
      f"(overall error rate {(~ok).mean() * 100:.1f}%).")