"""
finetune_oldrule.py -- finetune_maestro.py with exactly ONE change:
WindowDataset passes the "active pitch" (the pitch input of the rotary position embedding) with the rule
used BEFORE commit e35caa4 (18 Oct 2026): the last pitch seen. This is the rule the ARIA backbone was
pre-trained with. Everything else (arguments, data, model, optimiser, logging, W&B upload) is the
current finetune_maestro.py.

Purpose: an epoch-matched control for Section 4.3. Pre-training used the old rule, later fine-tuning used the
new one, so the data change (ARIA-MIDI -> MAESTRO) and the rule change happened together. If this run also
moves the pitch geometry from the circle of fifths towards the chromatic circle, the shift comes from MAESTRO.

Run from music-aria/src, with the same arguments as the fine-tuning run you want to repeat, e.g.
  python finetune_oldrule.py --init_ckpt <backbone.pt> --tag aria_init_oldrule --epochs 4 ...
"""
import os
import random

import numpy as np
import torch

import music_common as mc


def _old_rule_getitem(self, i):
    k = int(self.win_song[i])
    rel = int(self.win_rel[i])
    if self.jitter:
        rel = min(rel + 3 * random.randrange(self.hop // 3), int(self.max_rel[k]))
    start = int(self.song_start[k]) + rel
    pre = 2 if rel >= 3 else 0            # previous note's pitch token sits at rel-2
    arr = np.array(self.tokens[start - pre: start + self.seq_len + 1], dtype=np.int64)
    if self.time_stretch:
        arr = mc.stretch_ticks(arr, random.choice(mc.TIME_STRETCH_FACTORS))
    chunk = torch.from_numpy(arr)
    if self.transpose:
        shift = random.randint(-self.transpose, self.transpose)
        if shift != 0:
            is_p = chunk < 128
            if bool(is_p.any()):
                pv = chunk[is_p]
                if int(pv.min()) + shift >= 0 and int(pv.max()) + shift <= 127:
                    chunk[is_p] = pv + shift
    # --- OLD rule (before e35caa4): active pitch = last pitch token seen so far (60 before the first) ---
    is_p = chunk < 128
    ar = torch.arange(len(chunk))
    last_idx = torch.cummax(torch.where(is_p, ar, torch.zeros_like(ar)), dim=0).values
    seen = torch.cummax(is_p.long(), dim=0).values.bool()
    active = torch.where(seen, chunk[last_idx], torch.full_like(chunk, 60))
    chunk, active = chunk[pre:], active[pre:]
    return chunk[:-1], chunk[1:], active[:-1]


# Top level on purpose: with torch.multiprocessing.spawn every DDP worker re-imports this file as __mp_main__,
# so the patch is active in all workers, not only in the parent process.
mc.WindowDataset.__getitem__ = _old_rule_getitem

import finetune_maestro as fm  # noqa: E402  (imported after the patch)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    args = fm.parse_args()
    print(f"[finetune_oldrule] OLD active-pitch rule (pre-e35caa4) | tag {args.tag}", flush=True)
    mc.load_or_build_maestro_cache(args.maestro_dir, workers=os.cpu_count() or 4)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no GPU visible"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12356")
    if n_gpu > 1:
        mp.spawn(fm.main_worker, args=(n_gpu, args), nprocs=n_gpu, join=True)
    else:
        fm.main_worker(0, 1, args)
