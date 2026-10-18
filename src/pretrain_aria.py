"""
pretrain_aria.py  --  time-budgeted, resumable DDP pretraining on the ARIA memmap corpus (Kaggle T4 x2 friendly).

    python pretrain_aria.py --corpus_dir /kaggle/input/aria-corpus --out_dir /kaggle/working/artifacts --hours 9.5

Design for a 12 h session
  * The cosine LR schedule is driven by ELAPSED WALL-CLOCK / --hours (rank 0 broadcasts it every optimizer
    step), so training always ends exactly at the budget with the LR fully decayed, whatever the throughput.
  * Full resume (model, optimizer, scaler, schedule clock, data position): just re-run the same command.
  * Validation perplexity on held-out ARIA files every --eval_every steps; W&B logging + model artifacts.
  * Output: <out_dir>/aria_pretrained_backbone.pt  (model weights + model config) -> used by finetune_maestro.py
"""
import argparse
import math
import os
import pathlib
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data as data
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from music_common import (Logger, ShardedSampler, TypeStats, WindowDataset, add_model_args, assert_grammar,
                          build_model, cfg_from_args, count_params, evaluate, flatten_summary, format_summary,
                          init_dist, load_corpus, load_weights, lr_at, make_optimizer, masked_ce, save_atomic)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", default="", help="local corpus folder (also the download target for "
                   "--corpus_artifact)")
    p.add_argument("--corpus_artifact", default="", help="W&B artifact path, e.g. "
                   "riya-rajbhut-student/music-rnn-ddp/aria_corpus:latest")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--hours", type=float, default=9.5, help="wall-clock training budget (leave >=1.5 h of margin)")
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--hop", type=int, default=768, help="window stride; = seq_len -> no overlap")
    p.add_argument("--batch_size", type=int, default=16, help="per GPU micro-batch")
    p.add_argument("--accum", type=int, default=2, help="grad accumulation (tokens/step = 2 GPUs*16*2*768 ~ 49k)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--transpose", type=int, default=5, help="max random transposition (semitones)")
    p.add_argument("--time_stretch", action="store_true", help="augmentation: random time stretch 0.95-1.05")
    p.add_argument("--val_songs", type=int, default=300)
    p.add_argument("--val_windows", type=int, default=3072)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--upload_every_min", type=float, default=90.0, help="min minutes between W&B model uploads")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps (0 = off)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing pretrain_last.pt")
    p.add_argument("--init_ckpt", default="", help="start from these backbone weights (model + config) with a "
                   "NEW optimizer/schedule, e.g. to continue pretraining or extend the context window")
    p.add_argument("--run_name", default="aria-pretrain", help="W&B run name")
    p.add_argument("--artifact_prefix", default="aria_pretrained",
                   help="names of saved files / W&B artifacts: <prefix>_best, <prefix>_backbone")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="music-rnn-ddp")
    p.add_argument("--wandb_entity", default="riya-rajbhut-student")
    add_model_args(p)
    return p.parse_args()


def fetch_corpus(args):
    """Download the corpus from W&B once, in the parent process, before the workers are spawned."""
    if not args.corpus_artifact:
        assert args.corpus_dir, "give --corpus_dir or --corpus_artifact"
        return
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
    art = wandb.Api().artifact(args.corpus_artifact, type="dataset")
    args.corpus_dir = art.download(root=args.corpus_dir or "aria_corpus_dl")
    print(f"corpus downloaded from W&B artifact {args.corpus_artifact} -> {args.corpus_dir}", flush=True)


def main_worker(rank, world, args):
    init_dist(rank, world)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    torch.backends.cudnn.benchmark = True
    out_dir = pathlib.Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ------------------------------------------------------------------ data
    tokens, offsets, keep = load_corpus(args.corpus_dir)
    lens = np.diff(offsets)
    eligible = np.where(keep & (lens >= args.seq_len + 3))[0]
    perm = np.random.default_rng(args.seed).permutation(eligible)
    n_val = min(args.val_songs, max(20, int(0.01 * len(perm))), len(perm) // 2)
    val_ids, train_ids = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    train_ds = WindowDataset(tokens, offsets, train_ids, args.seq_len, args.hop, jitter=True, transpose=args.transpose,
                             time_stretch=args.time_stretch)
    val_ds = WindowDataset(tokens, offsets, val_ids, args.seq_len, args.seq_len, max_windows=args.val_windows)
    if is_main:
        print(f"corpus: {len(offsets) - 1} songs, {int(keep.sum())} kept after dedup | train songs {len(train_ids)} "
              f"({train_ds.n_tokens() / 1e6:.0f}M tokens/epoch, {len(train_ds)} windows) | val songs {len(val_ids)} "
              f"({len(val_ds)} windows)", flush=True)
    assert len(train_ds) // (world * args.batch_size) > 0, "corpus too small for this batch size"

    train_sampler = ShardedSampler(len(train_ds), rank, world, shuffle=True, seed=args.seed, drop_last=True)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=min(2, args.num_workers),
                                 sampler=ShardedSampler(len(val_ds), rank, world, shuffle=False), pin_memory=True)

    # ------------------------------------------------------------------ model / optimiser
    init_ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True) if args.init_ckpt else None
    cfg = init_ck["model_cfg"] if init_ck else cfg_from_args(args)
    torch.manual_seed(args.seed)                                   # identical init on every rank
    raw = build_model(cfg, dropout=args.dropout, grad_checkpoint=args.grad_checkpoint).to(device)
    if init_ck is not None:
        load_weights(raw, init_ck["model_state_dict"])
        if is_main:
            print(f"initialised from {args.init_ckpt} | meta {init_ck.get('meta')}", flush=True)
        del init_ck
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    model = DDP(raw, device_ids=[rank], broadcast_buffers=False)
    opt = make_optimizer(raw, args.lr, args.weight_decay)
    scaler = GradScaler("cuda")

    state = dict(step=0, epoch=0, micro=0, elapsed=0.0, best=float("inf"), wandb_id=None,
                 finished=False, last_upload=0.0)
    last_path = out_dir / "pretrain_last.pt"
    if last_path.exists() and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=True)
        raw.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        if is_main:
            print(f"RESUMED from {last_path}: step {state['step']}, epoch {state['epoch']}, "
                  f"{state['elapsed'] / 3600:.2f} h already used", flush=True)
    if is_main:
        print(f"model: {count_params(raw) / 1e6:.1f}M params | cfg {cfg}", flush=True)

    logger = Logger(is_main and not args.no_wandb, project=args.wandb_project, entity=args.wandb_entity,
                    name=args.run_name, config=vars(args), id=state["wandb_id"], resume="allow")
    if is_main:
        state["wandb_id"] = logger.run_id

    if state["finished"]:
        if is_main:
            print("Pretraining already finished for this out_dir (use --fresh to restart).", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------ helpers
    budget = args.hours * 3600.0
    elapsed_base = state["elapsed"]
    t0 = time.time()

    def elapsed():
        return elapsed_base + (time.time() - t0)

    step, epoch, micro = state["step"], state["epoch"], state["micro"]

    def save_last():
        state.update(step=step, epoch=epoch, micro=micro, elapsed=elapsed())
        save_atomic({"model_state_dict": raw.state_dict(), "optimizer": opt.state_dict(),
                     "scaler": scaler.state_dict(), "state": dict(state), "model_cfg": cfg}, last_path)

    def save_backbone(path, res):
        save_atomic({"model_state_dict": raw.state_dict(), "model_cfg": cfg,
                     "meta": dict(step=step, val_ppl=res["ppl"], val_nll=res["nll"], val_acc=res["acc"],
                                  hours=elapsed() / 3600)}, path)

    tr = TypeStats(device)
    tok_window, t_log = 0, time.time()
    tokens_per_step = world * args.batch_size * args.accum * args.seq_len
    grammar_checked = False
    done = False
    train_sampler.set_epoch(epoch, skip=micro * args.batch_size)

    if is_main:
        print(f"=== ARIA pretraining: budget {args.hours:.2f} h, {tokens_per_step} tokens/step ===", flush=True)

    # ------------------------------------------------------------------ training loop
    while not done:
        loader = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True,
                                 prefetch_factor=4 if args.num_workers > 0 else None)
        model.train()
        opt.zero_grad(set_to_none=True)
        for x, y, a in loader:
            x, y, a = x.to(device, non_blocking=True), y.to(device, non_blocking=True), a.to(device, non_blocking=True)
            if not grammar_checked:
                assert_grammar(x, y)
                grammar_checked = True
            last_micro = (micro + 1) % args.accum == 0
            with (nullcontext() if last_micro else model.no_sync()):
                with autocast("cuda", dtype=torch.float16):
                    logits, _ = model(x, a)
                loss, nll, pred = masked_ce(logits, y, x, args.label_smoothing)
                scaler.scale(loss / args.accum).backward()
            tr.update(pred, y, nll)
            micro += 1
            if not last_micro:
                continue

            # ---- optimizer step; LR follows the wall-clock budget (same value on every rank) ----
            prog_t = torch.tensor([elapsed() / budget], device=device, dtype=torch.float64)
            dist.broadcast(prog_t, 0)
            progress = float(prog_t.item())
            lr = lr_at(args.lr, step, args.warmup_steps, progress, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            scaler.unscale_(opt)
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), args.clip))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            tok_window += tokens_per_step

            if step % args.log_every == 0:
                tr.reduce()
                s = tr.summary()
                if not math.isfinite(s["nll"]):
                    raise RuntimeError("non-finite training loss")
                if is_main:
                    tps = tok_window / (time.time() - t_log)
                    print(f"[step {step}] ep {epoch} | {format_summary(s)} | lr {lr:.2e} | gnorm {gnorm:.2f} | "
                          f"{tps / 1e3:.1f}k tok/s | {elapsed() / 3600:.2f}/{args.hours:.2f} h", flush=True)
                    d = flatten_summary("train", s)
                    d.update({"lr": lr, "grad_norm": gnorm, "tokens_per_sec": tps, "progress": progress,
                              "epoch": epoch, "tokens_seen": step * tokens_per_step})
                    logger.log(d, step=step)
                tr.reset()
                tok_window, t_log = 0, time.time()

            if step % args.eval_every == 0:
                res = evaluate(raw, val_loader, device)
                model.train()
                if is_main:
                    print(f"[step {step}] VAL {format_summary(res)}", flush=True)
                    logger.log(flatten_summary("val", res), step=step)
                    if res["nll"] < state["best"]:
                        state["best"] = res["nll"]
                        save_backbone(out_dir / f"{args.artifact_prefix}_best.pt", res)
                        if (time.time() - state["last_upload"]) / 60 >= args.upload_every_min:
                            logger.log_model(out_dir / f"{args.artifact_prefix}_best.pt", f"{args.artifact_prefix}_best",
                                             dict(step=step, val_ppl=res["ppl"]))
                            state["last_upload"] = time.time()
                t_log = time.time()
                tok_window = 0

            if is_main and step % args.save_every == 0:
                save_last()

            if progress >= 1.0 or (args.max_steps > 0 and step >= args.max_steps):
                done = True
                break
        if not done:                                   # finished a pass over the corpus
            epoch += 1
            micro = 0
            train_sampler.set_epoch(epoch, skip=0)
            if is_main:
                print(f"=== completed pass {epoch} over the corpus ===", flush=True)

    # ------------------------------------------------------------------ finish
    res = evaluate(raw, val_loader, device)
    if is_main:
        print(f"[final] step {step} | VAL {format_summary(res)}", flush=True)
        logger.log(flatten_summary("val", res), step=step)
        final_path = out_dir / f"{args.artifact_prefix}_backbone.pt"
        save_backbone(final_path, res)
        logger.log_model(final_path, f"{args.artifact_prefix}_backbone", dict(step=step, val_ppl=res["ppl"], cfg=cfg))
        state["finished"] = True
        save_last()
        logger.summary({"final/val_ppl": res["ppl"], "final/val_acc": res["acc"], "final/steps": step})
        print(f"saved {final_path}", flush=True)
    dist.barrier()
    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    fetch_corpus(args)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no GPU visible"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    if n_gpu > 1:
        mp.spawn(main_worker, args=(n_gpu, args), nprocs=n_gpu, join=True)
    else:
        main_worker(0, 1, args)
