"""
prep_aria.py  --  turn a folder of ARIA-MIDI files into a uint16 memmap token corpus.

Run this in a CPU-only Kaggle session (no GPU quota used), then save /kaggle/working/aria_corpus as a
Kaggle Dataset and attach it to the pretraining session.

    python prep_aria.py --midi_dir /kaggle/input/aria-midi-pruned \
                        --out_dir  /kaggle/working/aria_corpus \
                        --max_tokens 600000000 --workers 4 --exclude_maestro_eval

Pipeline
  1. glob *.mid / *.midi, shuffle (seeded), optionally cap with --max_files
  2. (optional) quality score filter via --score_json {"<file stem>": score} and --min_score
  3. tokenise with the SAME tokenizer as MAESTRO (multiprocessing)
  4. heuristic quality filters computed from the tokens (too short, too dense, outside piano range, ...)
  5. stream tokens into tokens.u16  (stops early at --max_tokens)
  6. de-duplicate: exact hash + near-duplicate fingerprint overlap (keeps the first copy)
  7. (optional) drop every file that overlaps a MAESTRO validation/test composition (no leakage)
Outputs: tokens.u16, offsets.npy, keep.npy, sources.txt, report.json
Optional: --wandb_artifact aria_corpus  uploads the folder to W&B (stage in /tmp or /kaggle/working first;
          add --delete_local to remove the staging copy after a successful upload).
"""
import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from music_common import (MAX_TICKS, TICKS_PER_SECOND, fingerprint, load_or_build_maestro_cache,
                          midi_to_tokens)


def quality_reason(tokens, c):
    """Returns a rejection reason or None. All checks are computed from the token triplets."""
    n = len(tokens) // 3
    if n < c["min_notes"]:
        return "too_few_notes"
    shifts = tokens[0::3].astype(np.int64) - 128
    seconds = shifts.sum() / TICKS_PER_SECOND + (int(tokens[-1]) - 256) / TICKS_PER_SECOND
    if seconds < c["min_seconds"]:
        return "too_short"
    if n / max(seconds, 1e-6) > c["max_nps"]:
        return "too_dense"
    pitches = tokens[1::3]
    if ((pitches < 21) | (pitches > 108)).mean() > 0.02:
        return "out_of_piano_range"
    if (shifts[1:] >= MAX_TICKS).mean() > 0.10:
        return "long_silences"
    if ((tokens[2::3].astype(np.int64) - 256) <= 1).mean() > 0.30:
        return "broken_durations"
    return None


def process_file(job):
    idx, path, cfg = job
    tokens = midi_to_tokens(path)
    if tokens.size == 0:
        return idx, path, None, "unreadable_or_empty", None, None
    reason = quality_reason(tokens, cfg)
    if reason:
        return idx, path, None, reason, None, None
    return idx, path, tokens, None, fingerprint(tokens), hashlib.blake2b(tokens.tobytes(), digest_size=8).digest()


def find_near_duplicates(fps, n_main, frac=0.3, min_shared=3, cap=30):
    """fps: list of uint64 arrays; ids [0, n_main) are corpus songs, ids >= n_main are 'reference' songs
    (MAESTRO val/test) whose near-copies must be removed from the corpus.
    Returns (drop_mask over corpus songs, n_dropped_vs_reference, n_dropped_as_duplicate)."""
    sizes = np.array([len(f) for f in fps], dtype=np.int64)
    nonempty = [f for f in fps if len(f)]
    drop = np.zeros(n_main, dtype=bool)
    if not nonempty:
        return drop, 0, 0
    h = np.concatenate(fps)
    owner = np.repeat(np.arange(len(fps), dtype=np.int64), sizes)
    order = np.argsort(h, kind="stable")           # stable -> owners ascending inside equal-hash runs
    h, owner = h[order], owner[order]
    _, start, cnt = np.unique(h, return_index=True, return_counts=True)
    pairs = defaultdict(int)
    for s, c in zip(start[cnt >= 2], cnt[cnt >= 2]):
        if c > cap:                                # ultra-common n-grams (scales, arpeggios) carry no signal
            continue
        ids = owner[s:s + c].tolist()
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs[(ids[a], ids[b])] += 1

    def similar(i, j, c):
        return c >= max(min_shared, frac * min(sizes[i], sizes[j]))

    n_ref = n_dup = 0
    for (i, j), c in pairs.items():                # 1) overlap with reference (eval) compositions
        if i < n_main <= j and similar(i, j, c) and not drop[i]:
            drop[i] = True
            n_ref += 1
    for (i, j), c in sorted(pairs.items()):        # 2) near-duplicates inside the corpus: keep the earliest
        if j < n_main and not drop[i] and not drop[j] and similar(i, j, c):
            drop[j] = True
            n_dup += 1
    return drop, n_ref, n_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", nargs="+", required=True)
    ap.add_argument("--out_dir", default="aria_corpus")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--max_files", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max_tokens", type=int, default=0, help="stop once this many tokens were written (0 = no cap)")
    ap.add_argument("--min_notes", type=int, default=400)
    ap.add_argument("--min_seconds", type=float, default=30.0)
    ap.add_argument("--max_nps", type=float, default=25.0, help="max notes per second")
    ap.add_argument("--score_json", default="", help='optional {"file stem": quality score} json')
    ap.add_argument("--min_score", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--maestro_dir", default="data")
    ap.add_argument("--exclude_maestro_eval", action="store_true",
                    help="drop corpus files that overlap MAESTRO validation/test compositions")
    ap.add_argument("--wandb_artifact", default="", help="if set, upload the finished corpus as a W&B dataset "
                    "artifact with this name (e.g. aria_corpus)")
    ap.add_argument("--wandb_project", default="music-rnn-ddp")
    ap.add_argument("--wandb_entity", default="riya-rajbhut-student")
    ap.add_argument("--delete_local", action="store_true",
                    help="after a SUCCESSFUL W&B upload, delete the local staging folder")
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted({str(p) for d in a.midi_dir for pat in ("*.mid", "*.midi", "*.MID", "*.MIDI")
                    for p in pathlib.Path(d).rglob(pat)})
    print(f"found {len(paths)} MIDI files", flush=True)
    random.Random(a.seed).shuffle(paths)
    if a.score_json:
        scores = json.load(open(a.score_json))
        before = len(paths)
        paths = [p for p in paths if scores.get(pathlib.Path(p).stem, -1e9) >= a.min_score]
        print(f"score filter: {before} -> {len(paths)}", flush=True)
    if a.max_files > 0:
        paths = paths[:a.max_files]

    cfg = dict(min_notes=a.min_notes, min_seconds=a.min_seconds, max_nps=a.max_nps)
    jobs = ((i, p, cfg) for i, p in enumerate(paths))

    offsets, sources, fps, hashes = [0], [], [], []
    reasons = Counter()
    total_tokens = 0
    tok_path = out / "tokens.u16"
    with Pool(a.workers) as pool, open(tok_path, "wb") as f:
        for n_done, (idx, path, tokens, reason, fp, h) in enumerate(pool.imap(process_file, jobs, chunksize=8), 1):
            if tokens is None:
                reasons[reason] += 1
            else:
                tokens.tofile(f)
                total_tokens += len(tokens)
                offsets.append(total_tokens)
                sources.append(path)
                fps.append(fp)
                hashes.append(h)
            if n_done % 2000 == 0:
                print(f"  processed {n_done}/{len(paths)} | kept {len(sources)} | tokens {total_tokens / 1e6:.1f}M",
                      flush=True)
            if a.max_tokens > 0 and total_tokens >= a.max_tokens:
                print("reached --max_tokens, stopping early", flush=True)
                break

    n = len(sources)
    print(f"converted {n} songs, {total_tokens / 1e6:.1f}M tokens; rejected: {dict(reasons)}", flush=True)

    # --- exact duplicates (keep first) ---
    seen, exact_drop = set(), np.zeros(n, dtype=bool)
    for i, h in enumerate(hashes):
        if h in seen:
            exact_drop[i] = True
        seen.add(h)

    # --- near duplicates (+ optional MAESTRO eval overlap) ---
    ref_fps = []
    if a.exclude_maestro_eval:
        songs = load_or_build_maestro_cache(a.maestro_dir, workers=a.workers)
        ref_fps = [fingerprint(t) for sp in ("validation", "test") for t in songs[sp]]
        print(f"using {len(ref_fps)} MAESTRO validation/test songs as exclusion reference", flush=True)
    near_drop, n_ref, n_dup = find_near_duplicates(fps + ref_fps, n)

    keep = ~(exact_drop | near_drop)
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "keep.npy", keep)
    (out / "sources.txt").write_text("\n".join(sources))
    lens = np.diff(np.asarray(offsets))
    report = dict(files_found=len(paths), converted=n, rejected=dict(reasons), exact_duplicates=int(exact_drop.sum()),
                  near_duplicates=int(n_dup), dropped_overlap_with_maestro_eval=int(n_ref), kept_songs=int(keep.sum()),
                  kept_tokens=int(lens[keep].sum()), total_tokens_written=int(total_tokens))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    if a.wandb_artifact:
        upload_to_wandb(a, out, report)


def upload_to_wandb(a, out, report):
    """W&B cannot be written to directly, so the corpus is staged in --out_dir first and then uploaded.
    The local copy is only deleted (with --delete_local) after the upload has fully finished."""
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, job_type="prep-aria",
                     name="prep-aria", config=vars(a))
    print(f"[wandb] uploading corpus to {run.entity}/{run.project} as '{a.wandb_artifact}' ...", flush=True)
    art = wandb.Artifact(a.wandb_artifact, type="dataset", metadata=report,
                         description="ARIA-MIDI token corpus (uint16 memmap + offsets + keep mask)")
    art.add_dir(str(out))
    run.log_artifact(art).wait()
    print(f"[wandb] upload complete: {run.url}", flush=True)
    run.finish()
    if a.delete_local:
        shutil.rmtree(out)
        print(f"deleted local folder {out}", flush=True)


if __name__ == "__main__":
    main()
