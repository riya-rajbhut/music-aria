"""
finetune_maestro.py  --  fine-tune on MAESTRO (OFFICIAL composition-level split) and test.

  # fine-tune the ARIA-pretrained backbone
  python finetune_maestro.py --init_ckpt artifacts/aria_pretrained_backbone.pt --tag aria_init

  # baseline for a fair comparison: SAME architecture/recipe, random init
  python finetune_maestro.py --init_ckpt none --tag scratch

Logs true (unsmoothed) perplexity, per-token-type accuracy/perplexity, zero-shot validation of the
pretrained model before any MAESTRO training, final test metrics, and uploads the best model to W&B.
"""
import argparse
import math
import os
import pathlib
import random
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
                          build_model, cfg_from_args, concat_songs, count_params, evaluate, flatten_summary,
                          format_pitch_diag, format_position_curve, format_position_curve_by_type, format_summary,
                          format_tolerance, init_dist, load_or_build_maestro_cache, load_weights, lr_at,
                          make_optimizer, masked_ce, save_atomic)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_ckpt", default="artifacts/aria_pretrained_backbone.pt", help="path, or 'none' for scratch")
    p.add_argument("--tag", default="aria_init")
    p.add_argument("--maestro_dir", default="data")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--train_hop", type=int, default=384)
    p.add_argument("--eval_hop", type=int, default=384)
    p.add_argument("--no_jitter", action="store_true", help="fixed window grid (v9 behaviour)")
    p.add_argument("--score_from", type=int, default=0, help="evaluation only: count positions >= this index "
                   "(multiple of 3). Use with --eval_hop = seq_len - score_from so each token is scored once with "
                   ">= score_from tokens of context, e.g. --seq_len 768 --score_from 384 --eval_hop 384")
    p.add_argument("--tols", type=str, default="", help="evaluation: tolerance-accuracy levels in ticks for "
                   "time_shift/duration, e.g. 0,1,2,3 (1 tick = 31.25 ms). Reported IN ADDITION to exact accuracy.")
    p.add_argument("--transpose", type=int, default=5)
    p.add_argument("--time_stretch", action="store_true",
                   help="augmentation: random time stretch in {0.95,0.975,1,1.025,1.05} (MAESTRO-paper recipe)")
    p.add_argument("--batch_size", type=int, default=16, help="per GPU micro-batch")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--pitch_weight", type=float, default=1.0, help="training-loss weight on pitch targets")
    p.add_argument("--contrastive_weight", type=float, default=0.0, help="optional CPC aux loss (0 = off)")
    p.add_argument("--cpc_ks", type=str, default="3,6")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="no training: evaluate the weights given by --init_ckpt on MAESTRO val + test and exit")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="music-rnn-ddp")
    p.add_argument("--wandb_entity", default="riya-rajbhut-student")
    add_model_args(p)       # only used when --init_ckpt none (otherwise the checkpoint's own config is used)
    return p.parse_args()


def main_worker(rank, world, args):
    init_dist(rank, world)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    tols = tuple(int(t) for t in args.tols.split(",")) if args.tols else ()
    out_dir = pathlib.Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # ------------------------------------------------------------------ data (official split)
    songs = load_or_build_maestro_cache(args.maestro_dir, workers=1)    # cache was built in __main__
    sets = {}
    for split in ("train", "validation", "test"):
        tok, off = concat_songs(songs[split])
        sets[split] = (tok, off, np.arange(len(off) - 1))
    train_ds = WindowDataset(*sets["train"], args.seq_len, args.train_hop, jitter=not args.no_jitter,
                             transpose=args.transpose, time_stretch=args.time_stretch)
    val_ds = WindowDataset(*sets["validation"], args.seq_len, args.eval_hop)
    test_ds = WindowDataset(*sets["test"], args.seq_len, args.eval_hop)
    if is_main:
        print(f"MAESTRO official split | songs train/val/test = {len(songs['train'])}/{len(songs['validation'])}/"
              f"{len(songs['test'])} | windows = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}", flush=True)

    sampler = ShardedSampler(len(train_ds), rank, world, shuffle=True, seed=args.seed, drop_last=True)

    def eval_loader(ds):
        return data.DataLoader(ds, batch_size=args.batch_size * 2, num_workers=min(2, args.num_workers),
                               sampler=ShardedSampler(len(ds), rank, world, shuffle=False), pin_memory=True)

    val_loader, test_loader = eval_loader(val_ds), eval_loader(test_ds)

    # ------------------------------------------------------------------ model
    init = args.init_ckpt.lower() != "none"
    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True) if init else None
    cfg = ck["model_cfg"] if init else cfg_from_args(args)
    use_cpc = args.contrastive_weight > 0
    cpc = dict(proj_dim=128, ks=tuple(int(k) for k in args.cpc_ks.split(",")), tau=0.1, n_anchors=512) if use_cpc else None
    torch.manual_seed(args.seed)
    raw = build_model(cfg, dropout=args.dropout, grad_checkpoint=args.grad_checkpoint, cpc=cpc).to(device)
    if init:
        load_weights(raw, ck["model_state_dict"])
        if is_main:
            print(f"loaded pretrained weights from {args.init_ckpt} | meta {ck.get('meta')}", flush=True)
    init_meta = ck.get("meta") if init else None
    del ck
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if is_main:
        print(f"model: {count_params(raw) / 1e6:.1f}M params | cfg {cfg} | init={'ARIA' if init else 'random'}", flush=True)
    model = DDP(raw, device_ids=[rank], broadcast_buffers=False)
    opt = make_optimizer(raw, args.lr, args.weight_decay)
    scaler = GradScaler("cuda")

    best_path = out_dir / f"best_maestro_{args.tag}.pt"
    last_path = out_dir / f"finetune_last_{args.tag}.pt"
    st = dict(epoch=0, step=0, best=float("inf"), bad=0, wandb_id=None)
    if args.eval_only:        # e.g. --eval_only --init_ckpt artifacts/best_maestro_aria_init.pt
        v = evaluate(raw, val_loader, device, args.score_from, tols)
        t = evaluate(raw, test_loader, device, args.score_from, tols)
        if is_main and init_meta and "tag" in init_meta and args.score_from == 0:
            # a fine-tuned checkpoint stores the validation PPL measured at training time with this same protocol
            ok = abs(v["ppl"] / init_meta["val_ppl"] - 1.0) < 0.02
            print(f"[sanity] checkpoint meta val PPL {init_meta['val_ppl']:.3f} vs measured {v['ppl']:.3f} -> "
                  f"{'OK' if ok else 'MISMATCH: the model or the data path is not what was trained'}", flush=True)
        if is_main:
            extra = lambda r: (("\n" + format_tolerance(r)) if tols else "") + "\n" + format_pitch_diag(r)
            print(f"[eval_only] weights from {args.init_ckpt} | seq_len {args.seq_len}\n"
                  f"VAL  {format_summary(v)}\n{format_position_curve(v)}\n{format_position_curve_by_type(v)}{extra(v)}\n"
                  f"TEST {format_summary(t)}\n{format_position_curve(t)}\n{format_position_curve_by_type(t)}{extra(t)}",
                  flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if args.resume and last_path.exists():
        lk = torch.load(last_path, map_location=device, weights_only=True)
        raw.load_state_dict(lk["model_state_dict"])
        opt.load_state_dict(lk["optimizer"])
        scaler.load_state_dict(lk["scaler"])
        st.update(lk["state"])
        if is_main:
            print(f"RESUMED at epoch {st['epoch']}", flush=True)
    logger = Logger(is_main and not args.no_wandb, project=args.wandb_project, entity=args.wandb_entity,
                    name=f"maestro-{args.tag}", config=vars(args), id=st["wandb_id"], resume="allow")
    if is_main:
        st["wandb_id"] = logger.run_id

    # zero-shot validation of the pretrained model (transfer signal) -------------------------------
    if init and st["epoch"] == 0:
        r0 = evaluate(raw, val_loader, device, args.score_from)
        if is_main:
            print(f"[zero-shot, before MAESTRO training] VAL {format_summary(r0)}", flush=True)
            logger.log({**flatten_summary("val", r0), "epoch": 0}, step=0)

    # ------------------------------------------------------------------ training
    steps_per_epoch = (len(sampler) // args.batch_size) // args.accum
    total_steps = max(1, args.epochs * steps_per_epoch)
    step, grammar_checked = st["step"], False
    tr = TypeStats(device)
    for epoch in range(st["epoch"], args.epochs):
        sampler.set_epoch(epoch)
        loader = data.DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True,
                                 prefetch_factor=4 if args.num_workers > 0 else None)
        model.train()
        opt.zero_grad(set_to_none=True)
        tr.reset()
        micro, aux_sum, aux_n, gnorm, lr = 0, 0.0, 0, 0.0, args.lr
        for x, y, a in loader:
            x, y, a = x.to(device, non_blocking=True), y.to(device, non_blocking=True), a.to(device, non_blocking=True)
            if not grammar_checked:
                assert_grammar(x, y)
                grammar_checked = True
            last_micro = (micro + 1) % args.accum == 0
            with (nullcontext() if last_micro else model.no_sync()):
                with autocast("cuda", dtype=torch.float16):
                    logits, aux = model(x, a, targets=y if use_cpc else None)
                loss, nll, pred = masked_ce(logits, y, x, args.label_smoothing, args.pitch_weight)
                total = loss + (args.contrastive_weight * aux.float() if use_cpc else 0.0)
                scaler.scale(total / args.accum).backward()
            tr.update(pred, y, nll)
            if use_cpc:
                aux_sum, aux_n = aux_sum + float(aux.detach()), aux_n + 1
            micro += 1
            if not last_micro:
                continue
            lr = lr_at(args.lr, step, args.warmup_steps, step / total_steps, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            scaler.unscale_(opt)
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), args.clip))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if is_main and step % args.log_every == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] step {step} | loss {float(loss):.4f} | lr {lr:.2e} | "
                      f"gnorm {gnorm:.2f}", flush=True)

        tr.reduce()
        trs = tr.summary()
        if not math.isfinite(trs["nll"]):
            raise RuntimeError("non-finite training loss")
        val = evaluate(raw, val_loader, device, args.score_from)
        if is_main:
            print(f"Epoch [{epoch + 1}/{args.epochs}] | LR {lr:.2e} | TRAIN {format_summary(trs)}", flush=True)
            print(f"{'':>16}VAL   {format_summary(val)}", flush=True)
            d = {**flatten_summary("train", trs), **flatten_summary("val", val), "epoch": epoch + 1, "lr": lr}
            if use_cpc:
                d["train/aux_nce"] = aux_sum / max(aux_n, 1)
            logger.log(d, step=epoch + 1)

        improved = val["nll"] < st["best"]
        if improved:
            st["best"], st["bad"] = val["nll"], 0
            if is_main:
                save_atomic({"model_state_dict": raw.state_dict(), "model_cfg": cfg,
                             "meta": dict(epoch=epoch + 1, val_ppl=val["ppl"], val_acc=val["acc"], tag=args.tag)},
                            best_path)
                logger.log_model(best_path, f"maestro_best_{args.tag}",
                                 dict(epoch=epoch + 1, val_ppl=val["ppl"], val_acc=val["acc"]))
        else:
            st["bad"] += 1
        st.update(epoch=epoch + 1, step=step)
        if is_main:
            save_atomic({"model_state_dict": raw.state_dict(), "optimizer": opt.state_dict(),
                         "scaler": scaler.state_dict(), "state": dict(st), "model_cfg": cfg}, last_path)
        if st["bad"] >= args.patience:           # identical decision on all ranks (stats were all-reduced)
            if is_main:
                print(f"early stopping: no val improvement for {args.patience} epochs", flush=True)
            break

    # ------------------------------------------------------------------ test with best checkpoint
    dist.barrier()
    bk = torch.load(best_path, map_location=device, weights_only=True)
    raw.load_state_dict(bk["model_state_dict"])
    test = evaluate(raw, test_loader, device, args.score_from)
    if is_main:
        print(f"=== TEST (best checkpoint, epoch {bk['meta']['epoch']}) ===\n{format_summary(test)}\n"
              f"{format_position_curve(test)}", flush=True)
        logger.log(flatten_summary("test", test))
        logger.summary({"test/perplexity": test["ppl"], "test/accuracy": test["acc"],
                        "best_val_ppl": bk["meta"]["val_ppl"]})
    dist.barrier()
    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    load_or_build_maestro_cache(args.maestro_dir, workers=os.cpu_count() or 4)   # download + tokenise once
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no GPU visible"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12356")
    if n_gpu > 1:
        mp.spawn(main_worker, args=(n_gpu, args), nprocs=n_gpu, join=True)
    else:
        main_worker(0, 1, args)
