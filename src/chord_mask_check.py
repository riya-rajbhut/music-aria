"""
chord_mask_check.py  --  in the canonical chord order, a note that shares its tick with the previous note
(time shift == 0) can never have a lower pitch than the previous note. This script measures how much
probability the model still puts on such impossible pitches, and what masking them would change.

Usage (from the folder holding music_common.py):
  python chord_mask_check.py --canon --ckpt /kaggle/working/artifacts/best_maestro_canon.pt

Protocol: windows of 768 tokens, hop 384, notes in the second half of each window are scored (as in eval_ensemble.py).
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--canon", action="store_true", help="canonical chord order (CANON_CHORDS=1); needed for 'canon' checkpoints")
ap.add_argument("--maestro_dir", default="data")
args = ap.parse_args()
if args.canon:
    os.environ["CANON_CHORDS"] = "1"

import numpy as np
import torch
import torch.utils.data as data
from torch.amp import autocast

from music_common import WindowDataset, build_model, concat_songs, load_or_build_maestro_cache, load_weights

dev = torch.device("cuda:0")
N0 = 128                                    # first scored note (= score_from 384 / 3)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
model = build_model(ck["model_cfg"], dropout=0.0).to(dev)
load_weights(model, ck["model_state_dict"])
model.eval()
songs = load_or_build_maestro_cache(args.maestro_dir, workers=4)
print(f"tokenisation: {'canonical chord order' if args.canon else 'original order'}")

for split in ("validation", "test"):
    tok, off = concat_songs(songs[split])
    ds = WindowDataset(tok, off, np.arange(len(off) - 1), 768, 384)
    n = n_chord = viol = ok0 = ok1 = 0
    nll0 = nll1 = mass = 0.0
    with torch.no_grad():
        for x, y, a in data.DataLoader(ds, batch_size=16, num_workers=2):
            x, y, a = x.to(dev), y.to(dev), a.to(dev)
            with autocast("cuda", dtype=torch.float16):
                logits, _ = model(x, a)
            lg = logits[:, 0::3, :128].float()[:, N0:]                 # pitch logits of every scored note
            tgt = y[:, 0::3][:, N0:]
            shift = (x[:, 0::3] - 128)[:, N0:]
            prev = a[:, 0::3][:, N0:]                                  # pitch of the previous note
            chord = shift == 0                                         # same tick as the previous note
            below = (torch.arange(128, device=dev) < prev[..., None]) & chord[..., None]   # impossible pitches
            lp0 = torch.log_softmax(lg, -1)
            lp1 = torch.log_softmax(lg.masked_fill(below, float("-inf")), -1)
            n += tgt.numel()
            n_chord += int(chord.sum())
            viol += int((chord & (tgt < prev)).sum())
            mass += float((lp0.exp() * below).sum())
            ok0 += int((lp0.argmax(-1) == tgt).sum())
            ok1 += int((lp1.argmax(-1) == tgt).sum())
            nll0 += float(-lp0.gather(-1, tgt[..., None]).sum())
            nll1 += float(-lp1.gather(-1, tgt[..., None]).sum())
    print(f"\n{split}: {n} scored notes | chord notes (shift 0): {n_chord / n * 100:.1f}%")
    print(f"  true pitch below the previous pitch inside a tick (should be ~0 in canonical order): {viol}")
    print(f"  mean probability on impossible pitches, over chord notes: {mass / max(n_chord, 1) * 100:.2f}%")
    print(f"  pitch accuracy {ok0 / n * 100:.2f}% -> {ok1 / n * 100:.2f}% with the mask | "
          f"pitch ppl {np.exp(nll0 / n):.3f} -> {np.exp(nll1 / n):.3f}")