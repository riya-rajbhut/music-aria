"""
music_common.py  --  shared code for the ARIA-pretrain -> MAESTRO-finetune pipeline (v11)

Contents
  1. Tokenizer (REMI-like triplets: time_shift, pitch, duration) + MAESTRO helpers (official split)
  2. Fingerprinting (near-duplicate detection, used by prep_aria.py)
  3. Datasets / samplers (uint16 memmap friendly, resumable, numpy-only window index)
  4. Model: musical-time + pitch + token-index RoPE, factored embeddings, optional CPC aux loss
  5. Loss with grammar masking, per-type stats (accuracy / true perplexity), evaluation
  6. Distributed / logging / checkpoint utilities
"""
import csv
import math
import os
import pathlib
import pickle
import random
import time
import warnings
import zipfile
from datetime import timedelta
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import pretty_midi as pm  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint as torch_ckpt  # noqa: E402
import torch.utils.data as data  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.hub import download_url_to_file  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
warnings.filterwarnings("ignore", category=UserWarning, module="pretty_midi")

VOCAB = 384
TICKS_PER_SECOND = 32
MAX_TICKS = 127
TOKEN_TYPE_NAMES = ("pitch", "time_shift", "duration")


# =============================================================================
# 1. TOKENIZER + MAESTRO HELPERS
# =============================================================================
# Every note -> 3 tokens:  [time_shift 128..255] [pitch 0..127] [duration 256..383]

def midi_to_tokens(path) -> np.ndarray:
    """Same tokenisation as v9/v10 (1/32 s ticks, shift & duration capped at 127 ticks).
    Uses the non-drum instrument with the most notes (identical to instruments[0] for MAESTRO)."""
    empty = np.zeros(0, dtype=np.uint16)
    try:
        midi = pm.PrettyMIDI(str(path))
    except Exception:
        return empty
    insts = [i for i in midi.instruments if (not i.is_drum) and len(i.notes) > 0]
    if not insts:
        return empty
    inst = max(insts, key=lambda i: len(i.notes))
    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    n = len(notes)
    start = np.fromiter((x.start for x in notes), dtype=np.float64, count=n)
    end = np.fromiter((x.end for x in notes), dtype=np.float64, count=n)
    pitch = np.fromiter((x.pitch for x in notes), dtype=np.int64, count=n)
    prev = np.concatenate(([start[0]], start[:-1]))
    step = np.minimum(((start - prev) * TICKS_PER_SECOND).astype(np.int64), MAX_TICKS)
    dur = np.clip(((end - start) * TICKS_PER_SECOND).astype(np.int64), 1, MAX_TICKS)
    out = np.empty(3 * n, dtype=np.uint16)
    out[0::3] = 128 + step
    out[1::3] = pitch
    out[2::3] = 256 + dur
    return out


def token_type_ids(tokens):
    """0 = pitch (<128), 1 = time_shift (128..255), 2 = duration (>=256). Works for torch and numpy."""
    return (tokens >= 128).long() + (tokens >= 256).long() if torch.is_tensor(tokens) \
        else (tokens >= 128).astype(np.int64) + (tokens >= 256).astype(np.int64)


MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def ensure_maestro(dest_dir="data") -> pathlib.Path:
    """Returns the folder containing maestro-v3.0.0.csv (downloads if needed).
    `dest_dir` may also directly be an already-extracted MAESTRO folder (e.g. a Kaggle input)."""
    dest = pathlib.Path(dest_dir)
    if (dest / "maestro-v3.0.0.csv").exists():
        return dest
    folder = dest / "maestro-v3.0.0"
    if (folder / "maestro-v3.0.0.csv").exists():
        return folder
    dest.mkdir(parents=True, exist_ok=True)
    zip_target = dest / "maestro-v3.0.0-midi.zip"
    print("Downloading MAESTRO ...", flush=True)
    download_url_to_file(MAESTRO_URL, str(zip_target), progress=True)
    with zipfile.ZipFile(zip_target, "r") as z:
        z.extractall(dest)
    return folder


def load_or_build_maestro_cache(dest_dir="data", workers=4) -> dict:
    """{'train': [uint16 arrays], 'validation': [...], 'test': [...]} using MAESTRO's OFFICIAL split
    (split by composition, so no composition appears in two splits)."""
    folder = ensure_maestro(dest_dir)
    cache = folder / "converted_tokens_v7_official_split.pkl"
    if cache.exists():
        try:
            with cache.open("rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink()
    with open(folder / "maestro-v3.0.0.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = [str(folder / r["midi_filename"]) for r in rows]
    print(f"Tokenising {len(paths)} MAESTRO files with {workers} workers ...", flush=True)
    with Pool(workers) as pool:
        toks = pool.map(midi_to_tokens, paths, chunksize=4)
    songs = {"train": [], "validation": [], "test": []}
    for r, t in zip(rows, toks):
        if t.size >= 3:
            songs[r["split"]].append(t)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(songs, f)
    os.replace(tmp, cache)
    print({k: len(v) for k, v in songs.items()}, flush=True)
    return songs


def concat_songs(songs):
    """list of uint16 arrays -> (flat uint16 array, int64 offsets of length n+1)"""
    lens = np.array([len(s) for s in songs], dtype=np.int64)
    offsets = np.zeros(len(songs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    return np.concatenate(songs).astype(np.uint16), offsets


def load_corpus(corpus_dir):
    """Reads the corpus written by prep_aria.py -> (uint16 memmap, offsets, keep_mask)."""
    d = pathlib.Path(corpus_dir)
    tokens = np.memmap(d / "tokens.u16", dtype=np.uint16, mode="r")
    offsets = np.load(d / "offsets.npy")
    keep = np.load(d / "keep.npy") if (d / "keep.npy").exists() else np.ones(len(offsets) - 1, dtype=bool)
    return tokens, offsets, keep


# =============================================================================
# 2. FINGERPRINTS (transposition-invariant, offset-robust near-duplicate detection)
# =============================================================================
FP_NGRAM = 8      # intervals per n-gram
FP_MOD = 32       # keep ~1/32 of n-gram hashes


def fingerprint(tokens: np.ndarray) -> np.ndarray:
    """Set of sampled hashes of 8-interval n-grams of the pitch sequence. Two performances of the same
    piece share many of them even when transposed, transcribed with small errors, or cut at different points."""
    pitches = tokens[1::3].astype(np.int64)
    if len(pitches) < FP_NGRAM + 2:
        return np.zeros(0, dtype=np.uint64)
    iv = np.clip(np.diff(pitches), -24, 24) + 24
    m = len(iv) - FP_NGRAM + 1
    win = np.stack([iv[j:j + m] for j in range(FP_NGRAM)], axis=0)
    h = np.zeros(m, dtype=np.uint64)
    for j in range(FP_NGRAM):
        h = h * np.uint64(49) + win[j].astype(np.uint64)
    h = h * np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(29)
    nonconst = win.max(axis=0) != win.min(axis=0)       # ignore repeated-note / constant-interval runs
    sel = nonconst & ((h % np.uint64(FP_MOD)) == 0)
    return np.unique(h[sel])


# =============================================================================
# 3. DATASETS / SAMPLERS
# =============================================================================

class WindowDataset(data.Dataset):
    """
    Fixed-length windows over a flat uint16 token array (np.ndarray or np.memmap) + song offsets.
      * window index is held in numpy arrays (cheap for millions of windows)
      * every window starts on a time_shift token (start offsets are multiples of 3 within a song)
      * jitter=True : random 3-aligned start offset inside each hop (a different view every epoch)
      * transpose=k : random pitch shift in [-k, k] semitones (skipped if it would leave 0..127)
      * 'active pitch' (pitch of the most recent pitch token) is seeded from the previous note when the
        window starts mid-song, exactly like the full-song forward fill used in v9
    """

    def __init__(self, tokens, offsets, song_ids, seq_len, hop, jitter=False, transpose=0,
                 max_windows=None, subsample_seed=0):
        assert hop % 3 == 0 and seq_len % 3 == 0, "seq_len and hop must be multiples of 3"
        self.tokens, self.seq_len, self.hop = tokens, seq_len, hop
        self.jitter, self.transpose = jitter, transpose
        song_ids = np.asarray(song_ids, dtype=np.int64)
        lens = offsets[song_ids + 1] - offsets[song_ids]
        ok = lens >= seq_len + 3
        song_ids, lens = song_ids[ok], lens[ok]
        self.song_start = offsets[song_ids].astype(np.int64)
        self.max_rel = ((lens - seq_len - 1) // 3) * 3          # largest valid 3-aligned relative start
        counts = self.max_rel // hop + 1
        total = int(counts.sum())
        self.win_song = np.repeat(np.arange(len(song_ids), dtype=np.int32), counts)
        first = np.cumsum(counts) - counts
        self.win_rel = (np.arange(total, dtype=np.int64) - np.repeat(first, counts)) * hop
        if max_windows is not None and total > max_windows:
            keep = np.sort(np.random.default_rng(subsample_seed).choice(total, max_windows, replace=False))
            self.win_song, self.win_rel = self.win_song[keep], self.win_rel[keep]

    def __len__(self):
        return len(self.win_song)

    def n_tokens(self):
        return len(self) * self.seq_len

    def __getitem__(self, i):
        k = int(self.win_song[i])
        rel = int(self.win_rel[i])
        if self.jitter:
            rel = min(rel + 3 * random.randrange(self.hop // 3), int(self.max_rel[k]))
        start = int(self.song_start[k]) + rel
        pre = 2 if rel >= 3 else 0            # previous note's pitch token sits at rel-2
        chunk = torch.from_numpy(np.asarray(self.tokens[start - pre: start + self.seq_len + 1],
                                            dtype=np.int64).copy())
        if self.transpose:
            shift = random.randint(-self.transpose, self.transpose)
            if shift != 0:
                is_p = chunk < 128
                if bool(is_p.any()):
                    pv = chunk[is_p]
                    if int(pv.min()) + shift >= 0 and int(pv.max()) + shift <= 127:
                        chunk[is_p] = pv + shift
        is_p = chunk < 128
        ar = torch.arange(len(chunk))
        last_idx = torch.cummax(torch.where(is_p, ar, torch.zeros_like(ar)), dim=0).values
        seen = torch.cummax(is_p.long(), dim=0).values.bool()
        active = torch.where(seen, chunk[last_idx], torch.full_like(chunk, 60))
        chunk, active = chunk[pre:], active[pre:]
        return chunk[:-1], chunk[1:], active[:-1]


class ShardedSampler(data.Sampler):
    """Rank-sharded sampler with deterministic per-epoch shuffle and an optional `skip` (for exact resume).
    shuffle=False + drop_last=False gives an exact (un-padded) evaluation shard."""

    def __init__(self, n, rank, world, shuffle, seed=0, drop_last=False):
        # no super().__init__(): Sampler.__init__ changed across torch versions (newer ones have none)
        self.n, self.rank, self.world = int(n), rank, world
        self.shuffle, self.seed, self.drop_last = shuffle, seed, drop_last
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def _order(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.n) if self.shuffle \
            else np.arange(self.n)
        if self.drop_last:
            order = order[: (self.n // self.world) * self.world]
        return order[self.rank::self.world][self.skip:]

    def __iter__(self):
        for i in self._order():
            yield int(i)

    def __len__(self):
        m = (self.n // self.world) * self.world if self.drop_last else self.n
        return max(0, len(range(self.rank, m, self.world)) - self.skip)


# =============================================================================
# 4. MODEL
# =============================================================================

def musical_ticks(tokens):
    """(B,S) tokens -> (B,S) cumulative musical time in ticks (1/32 s). Chord tokens share the same time."""
    is_shift = (tokens >= 128) & (tokens < 256)
    return torch.where(is_shift, tokens - 128, torch.zeros_like(tokens)).cumsum(dim=1)


class MusicRoPE(nn.Module):
    """
    Rotary embedding whose head dims are split into three groups:
      dims[0] : REAL musical time   (cumulative ticks)      -> relative onset distance
      dims[1] : pitch               (active pitch)          -> relative pitch interval
      dims[2] : token index                                  -> order among tokens at the same onset
    Angles are computed directly (no lookup table, no clamp) once per forward and shared by all layers.
    Use dims=(32,32,0) to ablate the token-index group.
    """
    BASES = (10000.0, 10000.0, 1000.0)

    def __init__(self, head_dim, dims):
        super().__init__()
        dims = tuple(int(d) for d in dims)
        assert len(dims) == 3 and sum(dims) == head_dim and all(d % 2 == 0 for d in dims), \
            f"rope_dims {dims} must be 3 even numbers summing to head_dim={head_dim}"
        self.dims = dims
        for i, (d, b) in enumerate(zip(dims, self.BASES)):
            inv = 1.0 / (b ** (torch.arange(0, d, 2).float() / d)) if d > 0 else torch.zeros(0)
            self.register_buffer(f"inv{i}", inv, persistent=False)

    @property
    def active_dims(self):
        return tuple(d for d in self.dims if d > 0)

    def build(self, tokens, pitch_seq):
        B, S = tokens.shape
        pos = (musical_ticks(tokens).float(), pitch_seq.float(),
               torch.arange(S, device=tokens.device, dtype=torch.float32).expand(B, S))
        cs = []
        for i, p in enumerate(pos):
            if self.dims[i] == 0:
                continue
            ang = p.unsqueeze(-1) * getattr(self, f"inv{i}")
            ang = torch.cat([ang, ang], dim=-1).unsqueeze(1)          # (B,1,S,d)
            cs.append((ang.cos(), ang.sin()))
        return cs


def apply_rope(x, cs, dims):
    outs = []
    for part, (cos, sin), d in zip(x.split(dims, dim=-1), cs, dims):
        h = d // 2
        rot = torch.cat((-part[..., h:], part[..., :h]), dim=-1)
        outs.append(part * cos.to(part.dtype) + rot * sin.to(part.dtype))
    return torch.cat(outs, dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, rope_dims):
        super().__init__()
        self.h, self.dh, self.p = heads, dim // heads, dropout
        self.rope_dims = rope_dims
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.resid = nn.Dropout(dropout)

    def forward(self, x, cs):
        B, S, C = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, cs, self.rope_dims), apply_rope(k, cs, self.rope_dims)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.p if self.training else 0.0)
        return self.resid(self.proj(out.transpose(1, 2).reshape(B, S, C)))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout, rope_dims):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, rope_dims)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, cs):
        x = x + self.attn(self.ln1(x), cs)
        return x + self.mlp(self.ln2(x))


class TokenEmbedding(nn.Module):
    """Optionally factored: W[pitch p] = free[p] + pitch_class[p % 12] + octave[p // 12]. The assembled
    matrix is used for BOTH the lookup and the tied output projection (computed once per forward)."""

    def __init__(self, num_tokens, dim, factored):
        super().__init__()
        self.factored = factored
        self.free = nn.Embedding(num_tokens, dim)
        if factored:
            self.pitch_class = nn.Embedding(12, dim)
            self.octave = nn.Embedding(11, dim)
            p = torch.arange(128)
            self.register_buffer("pc_idx", p % 12, persistent=False)
            self.register_buffer("oct_idx", p // 12, persistent=False)

    def weight(self):
        w = self.free.weight
        if not self.factored:
            return w
        shared = self.pitch_class(self.pc_idx) + self.octave(self.oct_idx)
        return torch.cat([w[:128] + shared, w[128:]], dim=0)


class LookaheadInfoNCE(nn.Module):
    """Optional CPC-style auxiliary loss (off by default). Positive = token k steps ahead; same-id
    in-batch candidates are masked as false negatives."""

    def __init__(self, dim, proj_dim=128, ks=(3, 6), tau=0.1, n_anchors=512):
        super().__init__()
        self.ks, self.tau, self.n_anchors = tuple(ks), tau, n_anchors
        self.ctx_proj = nn.ModuleList([nn.Linear(dim, proj_dim, bias=False) for _ in self.ks])
        self.key_proj = nn.Linear(dim, proj_dim, bias=False)

    def forward(self, h, emb, targets):
        B, S, D = h.shape
        total, count = h.new_zeros((), dtype=torch.float32), 0
        with autocast("cuda", enabled=False):
            h, emb = h.float(), emb.float()
            for k, proj in zip(self.ks, self.ctx_proj):
                if k >= S:
                    continue
                c = h[:, : S - (k - 1)].reshape(-1, D)
                z = emb[:, k - 1:].reshape(-1, D)
                y = targets[:, k - 1:].reshape(-1)
                n = min(self.n_anchors, c.size(0))
                idx = torch.randperm(c.size(0), device=c.device)[:n]
                c = F.normalize(proj(c[idx]), dim=-1)
                z = F.normalize(self.key_proj(z[idx]), dim=-1)
                y = y[idx]
                logits = (c @ z.t()) / self.tau
                same = y[:, None] == y[None, :]
                eye = torch.eye(n, dtype=torch.bool, device=y.device)
                logits = logits.masked_fill(same & ~eye, float("-inf"))
                total = total + F.cross_entropy(logits, torch.arange(n, device=y.device))
                count += 1
        return total / max(count, 1)


class MusicTransformer(nn.Module):
    def __init__(self, num_tokens=VOCAB, embed_dim=768, num_heads=12, num_layers=12, dropout=0.1,
                 factored=True, rope_dims=(32, 16, 16), cpc=None, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.rope = MusicRoPE(embed_dim // num_heads, rope_dims)
        self.embed = TokenEmbedding(num_tokens, embed_dim, factored)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, dropout, self.rope.active_dims)
                                     for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.cpc = LookaheadInfoNCE(embed_dim, **cpc) if cpc else None
        self.apply(self._init)
        if factored:
            nn.init.normal_(self.embed.pitch_class.weight, 0.0, 0.01)
            nn.init.normal_(self.embed.octave.weight, 0.0, 0.01)
        res_std = 0.02 / math.sqrt(2 * num_layers)          # GPT-2 style scaled residual init
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, 0.0, res_std)
            nn.init.normal_(b.mlp[2].weight, 0.0, res_std)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, tokens, pitch_seq, targets=None):
        W = self.embed.weight()
        x = self.drop(F.embedding(tokens, W))
        cs = self.rope.build(tokens, pitch_seq)            # computed ONCE, shared by every layer
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch_ckpt.checkpoint(blk, x, cs, use_reentrant=False)
            else:
                x = blk(x, cs)
        x = self.ln_f(x)
        logits = F.linear(x, W)                            # tied output head
        aux = self.cpc(x, F.embedding(targets, W), targets) if (self.cpc is not None and targets is not None) else None
        return logits, aux


def add_model_args(p):
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--rope_dims", type=str, default="32,16,16", help="time,pitch,token-index dims; sum = head_dim")
    p.add_argument("--no_factored", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true", help="save memory (slower) if you hit OOM on a T4")


def cfg_from_args(a):
    return dict(num_tokens=VOCAB, embed_dim=a.embed_dim, num_heads=a.num_heads, num_layers=a.num_layers,
                factored=not a.no_factored, rope_dims=[int(x) for x in a.rope_dims.split(",")])


def build_model(cfg, dropout, grad_checkpoint=False, cpc=None):
    return MusicTransformer(dropout=dropout, grad_checkpoint=grad_checkpoint, cpc=cpc, **cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 5. LOSS WITH GRAMMAR MASK, STATS, EVALUATION
# =============================================================================
_TABLES = {}


def _tables(device):
    key = str(device)
    if key not in _TABLES:
        allowed = torch.zeros(3, VOCAB, dtype=torch.bool)
        allowed[0, :128] = True
        allowed[1, 128:256] = True
        allowed[2, 256:] = True
        nxt = torch.tensor([2, 0, 1])      # input type -> next type: pitch->duration, shift->pitch, duration->shift
        _TABLES[key] = (allowed.to(device), nxt.to(device))
    return _TABLES[key]


def assert_grammar(x, y):
    """Sanity check on the first batch: every target must have the token type implied by its input."""
    _, nxt = _tables(x.device)
    ok = token_type_ids(y) == nxt[token_type_ids(x)]
    if not bool(ok.all()):
        raise RuntimeError(f"Token grammar violated at {(~ok).float().mean().item() * 100:.2f}% of positions "
                           "(corpus is not a clean shift/pitch/duration cycle starting on a time_shift).")


def masked_ce(logits, targets, inputs, smoothing=0.0):
    """Cross-entropy over the VALID token type only (type is derived from the input token).
    Label smoothing is spread over valid classes only (nn.CrossEntropyLoss would give inf here).
    Returns (training loss, per-token UNSMOOTHED nll [detached], argmax prediction)."""
    allowed_tbl, nxt_tbl = _tables(inputs.device)
    allowed = allowed_tbl[nxt_tbl[token_type_ids(inputs)]]                 # (B,S,V)
    logp = F.log_softmax(logits.float().masked_fill(~allowed, float("-inf")), dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if smoothing > 0:
        smooth = -logp.masked_fill(~allowed, 0.0).sum(-1) / allowed.sum(-1)
        loss = ((1.0 - smoothing) * nll + smoothing * smooth).mean()
    else:
        loss = nll.mean()
    return loss, nll.detach(), logp.argmax(-1)


class TypeStats:
    """Accumulates correct / count / nll-sum per token type. PPL = exp(mean UNSMOOTHED masked NLL)."""

    def __init__(self, device):
        self.buf = torch.zeros(3, 3, dtype=torch.float64, device=device)

    def reset(self):
        self.buf.zero_()

    def update(self, pred, target, nll):
        tid = token_type_ids(target).reshape(-1)
        self.buf[0].index_add_(0, tid, (pred == target).reshape(-1).double())
        self.buf[1].index_add_(0, tid, torch.ones_like(tid, dtype=torch.float64))
        self.buf[2].index_add_(0, tid, nll.reshape(-1).double())

    def reduce(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.buf)

    def summary(self):
        b = self.buf.detach().cpu().numpy()
        tot = max(float(b[1].sum()), 1.0)
        nll = float(b[2].sum()) / tot
        out = dict(tokens=int(b[1].sum()), acc=float(b[0].sum()) / tot, nll=nll,
                   ppl=math.exp(min(nll, 50.0)), types={})
        for i, name in enumerate(TOKEN_TYPE_NAMES):
            c = max(float(b[1, i]), 1.0)
            n_i = float(b[2, i]) / c
            out["types"][name] = dict(acc=float(b[0, i]) / c, nll=n_i, ppl=math.exp(min(n_i, 50.0)))
        return out


def format_summary(s):
    t = " / ".join(f"{k} {v['acc'] * 100:.1f}% (ppl {v['ppl']:.2f})" for k, v in s["types"].items())
    return f"NLL {s['nll']:.4f} | PPL {s['ppl']:.3f} | Acc {s['acc'] * 100:.2f}% | {t}"


def flatten_summary(prefix, s):
    d = {f"{prefix}/nll": s["nll"], f"{prefix}/perplexity": s["ppl"], f"{prefix}/accuracy": s["acc"]}
    for k, v in s["types"].items():
        d[f"{prefix}/acc_{k}"] = v["acc"]
        d[f"{prefix}/ppl_{k}"] = v["ppl"]
    return d


@torch.no_grad()
def evaluate(raw_model, loader, device):
    """Evaluate on this rank's shard, then all-reduce. Uses the UNWRAPPED model (no DDP collectives), so
    ranks may have different numbers of batches. Caller must call model.train() afterwards."""
    raw_model.eval()
    st = TypeStats(device)
    for x, y, a in loader:
        x, y, a = x.to(device, non_blocking=True), y.to(device, non_blocking=True), a.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16):
            logits, _ = raw_model(x, a)
        _, nll, pred = masked_ce(logits, y, x)
        st.update(pred, y, nll)
    st.reduce()
    return st.summary()


# =============================================================================
# 6. DISTRIBUTED / LOGGING / CHECKPOINT UTILITIES
# =============================================================================

def init_dist(rank, world, timeout_hours=2):
    torch.cuda.set_device(rank)                      # before init: avoids the NCCL rank->GPU mapping warning
    dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank,
                            timeout=timedelta(hours=timeout_hours))


def lr_at(peak, step, warmup_steps, progress, min_ratio):
    """Linear warm-up (by steps) x cosine decay (by `progress` in [0,1]) down to peak*min_ratio."""
    warm = min(1.0, (step + 1) / max(1, warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return peak * warm * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


def save_atomic(obj, path):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_weights(model, state_dict):
    """Load a (possibly CPC-less) backbone checkpoint; only 'cpc.*' keys may be missing."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad = [k for k in missing if not k.startswith("cpc.")]
    if unexpected or bad:
        raise RuntimeError(f"Checkpoint mismatch. missing={bad} unexpected={list(unexpected)}")


class Logger:
    """Thin W&B wrapper (rank 0 only). Reads WANDB_API_KEY from the environment if set (e.g. Kaggle Secrets);
    silently disables itself if W&B is unavailable so training never crashes because of logging."""

    def __init__(self, enabled, **init_kwargs):
        self.run, self.run_id, self._wandb = None, None, None
        if not enabled:
            return
        try:
            import wandb
            key = os.environ.get("WANDB_API_KEY")
            if key:
                wandb.login(key=key)
            self.run = wandb.init(**init_kwargs)
            self._wandb, self.run_id = wandb, self.run.id
        except Exception as e:  # noqa: BLE001
            print(f"[wandb disabled] {e}", flush=True)
            self.run = None

    def log(self, d, step=None):
        if self.run is not None:
            self.run.log(d, step=step)

    def summary(self, d):
        if self.run is not None:
            for k, v in d.items():
                self.run.summary[k] = v

    def log_model(self, path, name, metadata=None):
        if self.run is None:
            return
        try:
            art = self._wandb.Artifact(name, type="model", metadata=metadata or {})
            art.add_file(str(path))
            self.run.log_artifact(art)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb artifact upload failed] {e}", flush=True)

    def finish(self):
        if self.run is not None:
            self.run.finish()
