#!/usr/bin/env python3
"""
Single-file Griffin-style trainer with staged shadow-stream training.

This entrypoint keeps the existing training/shadow pipeline intact while
swapping the Transformer backbone for a practical Griffin-style backbone:

1. train backbone
2. freeze backbone
3. train shadow stream on top
4. evaluate jointly

Design goals:
- minimal disruption to the existing codebase
- high reuse of the stable data / eval / checkpoint / shadow utilities
- simple, trainable Griffin-style hybrid of local attention + gated recurrence
- shadow-stream compatibility via the same forward/control contract

Cheap smoke test commands:

Backbone train smoke:
```bash
python3 train_griffin.py \
  --mode backbone \
  --device auto \
  --data-path ./data/datasets/fineweb10B_sp1024 \
  --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
  --block-size 128 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --local-window 32 \
  --dropout 0.0 \
  --train-batch-tokens 4096 \
  --val-batch-tokens 4096 \
  --max-steps 4 \
  --eval-interval 2 \
  --log-interval 1 \
  --save-every 0 \
  --max-val-batches 2 \
  --checkpoint-path checkpoints/griffin_smoke_{step}.pt
```

Eval smoke:
```bash
python3 train_griffin.py \
  --mode backbone \
  --device auto \
  --data-path ./data/datasets/fineweb10B_sp1024 \
  --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
  --block-size 128 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 128 \
  --local-window 32 \
  --dropout 0.0 \
  --val-batch-tokens 4096 \
  --max-val-batches 2 \
  --load-checkpoint checkpoints/griffin_smoke_4.pt \
  --eval-only
```
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    import zstandard

    _COMPRESSOR = "zstd"
except ImportError:
    zstandard = None
    _COMPRESSOR = "zlib"

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    try:
        from flash_attn_interface import flash_attn_func as _flash_attn_func
    except Exception:
        _flash_attn_func = None


@dataclass
class ShadowConfig:
    enabled: bool = True
    window: int = 24
    hidden_dim: int = 128
    dropout: float = 0.1
    control_scale: float = 0.14
    gate_bias: float = -1.5
    gate_l1_coef: float = 0.01
    gate_threshold: float = 0.55
    gate_sharpness: float = 12.0
    active_target: float = 0.18
    active_excess_coef: float = 0.03
    regime_temp: float = 0.7
    boundary_scale: float = 0.08
    boundary_soft_threshold: float = 0.72
    boundary_hard_threshold: float = 0.90
    suppression_scale: float = 0.18
    suppression_floor: float = 0.78
    suppression_l1_coef: float = 0.005
    token_dropout_max: float = 0.08
    corruption_prob: float = 0.05
    fingerprint_buckets: int = 64
    fingerprint_scale: float = 0.05
    recurrence_scale: float = 0.05
    pos_warp_scale: float = 0.05
    adapter_scale: float = 0.08
    adapter_layers: int = 2
    reset_scale: float = 0.06
    logit_scale: float = 0.04
    consistency_coef: float = 0.05
    template_slots: int = 4
    template_scale: float = 0.08
    ancestry_scale: float = 0.05
    ancestry_coef: float = 0.05
    ln_mod_scale: float = 0.08
    template_norm_coef: float = 1e-4
    adapter_norm_coef: float = 5e-5
    logit_norm_coef: float = 5e-5
    ln_norm_coef: float = 5e-5
    ancestry_entropy_coef: float = 0.02
    slot_balance_coef: float = 0.02


@dataclass
class TrainConfig:
    mode: str = "backbone"
    device_preference: str = "auto"
    seed: int = 1337
    run_id: str = "run"
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    train_pattern: str = ""
    val_pattern: str = ""
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    train_batch_tokens: int = 131072
    val_batch_tokens: int = 131072
    max_steps: int = 2000
    warmup_steps: int = 50
    eval_interval: int = 200
    log_interval: int = 20
    save_every: int = 500
    max_val_batches: int = 8
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    backbone_lr: float = 3e-4
    shadow_lr: float = 7e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    backbone_optimizer: str = "adamw"
    shadow_optimizer: str = "adamw"
    backbone_embed_lr_scale: float = 0.5
    backbone_scalar_lr_scale: float = 0.25
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    muon_nesterov: bool = True
    checkpoint_path: str = ""
    load_checkpoint: str = ""
    eval_only: bool = False
    report_bpb: bool = False
    legacy_export_path: str = ""
    metrics_csv_path: str = ""
    export_calibration_batches: int = 4
    export_max_loss_increase: float = 0.01
    export_probe_every: int = 0
    matrix_qat_start_step: int = 0
    qat_start_step: int = 0
    generate_tokens: int = 0
    compile_model: bool = False

    def resolved_train_pattern(self) -> str:
        return self.train_pattern or os.path.join(self.data_path, "fineweb_train_*.bin")

    def resolved_val_pattern(self) -> str:
        return self.val_pattern or os.path.join(self.data_path, "fineweb_val_*.bin")


@dataclass
class ExperimentConfig:
    model: Any
    shadow: ShadowConfig
    train: TrainConfig


class TeeStdout:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def append_metrics_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{name.replace("-", "_"): default})


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def global_rank() -> int:
    return dist.get_rank() if distributed_is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if distributed_is_initialized() else 1


def is_master_process() -> bool:
    return global_rank() == 0


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if is_master_process():
        print(*args, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = getattr(model, "_orig_mod")
            continue
        return model


def get_device(preference: str = "auto") -> torch.device:
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False.")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_default_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        major, _ = torch.cuda.get_device_capability(device)
        return torch.float16 if major >= 7 else torch.float32
    return torch.float32


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(x, device) for x in batch)
    return batch


def autocast_context(device: torch.device, amp_dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def supports_fast_attention(device: torch.device, use_flash_flag: bool) -> bool:
    return bool(use_flash_flag and device.type == "cuda" and hasattr(F, "scaled_dot_product_attention"))


def log_device_info(device: torch.device, amp_dtype: torch.dtype, use_flash: bool) -> None:
    print(f"device:{device} amp_dtype:{amp_dtype} fast_attention:{use_flash}")
    if device.type == "cuda":
        print(f"cuda_name:{torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("mps_path: enabled standard PyTorch kernels; CUDA-only fused paths are disabled")
    else:
        print("cpu_path: using safe eager PyTorch kernels")


@contextlib.contextmanager
def setup_run_logging(run_id: str) -> Any:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.txt"
    print(log_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        sys.stdout = TeeStdout(original_stdout, log_file)
        sys.stderr = TeeStdout(original_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens.astype(np.int64, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance()
                continue
            chunk_len = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + chunk_len])
            self.pos += chunk_len
            remaining -= chunk_len
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


class StreamingTokenLoader:
    def __init__(self, pattern: str, rank: int = 0, world_size: int = 1):
        self.stream = TokenStream(pattern)
        self.rank = rank
        self.world_size = max(1, int(world_size))

    def next_batch(self, batch_tokens: int, block_size: int) -> dict[str, Tensor]:
        usable = (batch_tokens // (block_size * self.world_size)) * (block_size * self.world_size)
        if usable <= 0:
            raise ValueError(
                f"Batch token budget too small for block_size={block_size} and world_size={self.world_size}"
            )
        local_tokens = usable // self.world_size
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size).long()
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span]
        x = local[:-1].view(-1, block_size)
        y = local[1:].view(-1, block_size)
        return {"input_ids": x, "targets": y}


def load_validation_tokens(pattern: str, block_size: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files], dim=0).contiguous().long()
    usable = ((tokens.numel() - 1) // block_size) * block_size
    if usable <= 0:
        raise ValueError(f"Validation split too short for block_size={block_size}")
    return tokens[: usable + 1]


def iter_validation_batches(val_tokens: Tensor, batch_tokens: int, block_size: int, max_batches: int | None = None) -> list[dict[str, Tensor]]:
    usable = (batch_tokens // block_size) * block_size
    if usable <= 0:
        raise ValueError(f"VAL_BATCH_TOKENS too small for block_size={block_size}")
    batch_seqs = usable // block_size
    total_seqs = (val_tokens.numel() - 1) // block_size
    batches: list[dict[str, Tensor]] = []
    for i, seq_start in enumerate(range(0, total_seqs, batch_seqs)):
        if max_batches is not None and i >= max_batches:
            break
        seq_end = min(seq_start + batch_seqs, total_seqs)
        raw = val_tokens[seq_start * block_size : seq_end * block_size + 1]
        x = raw[:-1].view(-1, block_size)
        y = raw[1:].view(-1, block_size)
        batches.append({"input_ids": x, "targets": y})
    return batches


def build_sentencepiece_luts(tokenizer_path: str, vocab_size: int) -> tuple[Tensor, Tensor, Tensor] | None:
    if spm is None or not tokenizer_path or not Path(tokenizer_path).is_file():
        return None
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return (
        torch.from_numpy(base_bytes),
        torch.from_numpy(has_leading_space),
        torch.from_numpy(is_boundary),
    )


def build_shadow_vocab_metadata(tokenizer_path: str, vocab_size: int) -> dict[str, Tensor]:
    table_size = max(1, vocab_size)
    token_class = np.zeros((table_size,), dtype=np.int64)
    scalar_names = (
        "leading_space",
        "alpha_ratio",
        "digit_ratio",
        "punct_ratio",
        "upper_ratio",
        "titleish",
        "urlish",
        "markupish",
        "fieldish",
        "shortish",
    )
    scalars = {name: np.zeros((table_size,), dtype=np.float32) for name in scalar_names}
    if spm is None or not tokenizer_path or not Path(tokenizer_path).is_file():
        return {"token_class": torch.from_numpy(token_class), **{name: torch.from_numpy(values) for name, values in scalars.items()}}
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    table_size = max(table_size, int(sp.vocab_size()))
    if table_size != token_class.shape[0]:
        token_class = np.pad(token_class, (0, table_size - token_class.shape[0]))
        scalars = {name: np.pad(values, (0, table_size - values.shape[0])).astype(np.float32, copy=False) for name, values in scalars.items()}
    for token_id in range(min(int(sp.vocab_size()), table_size)):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        if sp.is_byte(token_id):
            token_class[token_id] = 6
            scalars["punct_ratio"][token_id] = 1.0
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            scalars["leading_space"][token_id] = 1.0
            piece = piece[1:]
        text = piece.replace("▁", " ").strip()
        length = max(len(text), 1)
        alpha = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        uppers = sum(ch.isupper() for ch in text)
        punct = sum(not ch.isalnum() and not ch.isspace() for ch in text)
        lower = text.lower()
        scalars["alpha_ratio"][token_id] = alpha / length
        scalars["digit_ratio"][token_id] = digits / length
        scalars["punct_ratio"][token_id] = punct / length
        scalars["upper_ratio"][token_id] = uppers / max(alpha, 1)
        scalars["titleish"][token_id] = float(bool(text[:1].isupper() and text[1:].islower() if len(text) > 1 else text[:1].isupper()))
        scalars["urlish"][token_id] = float(any(marker in lower for marker in ("http", "www", ".com", ".org", ".net", "://", "@", "/")))
        scalars["markupish"][token_id] = float(any(ch in "<>{}[]=|_" for ch in text))
        scalars["fieldish"][token_id] = float(any(ch in ":/=-" for ch in text))
        scalars["shortish"][token_id] = float(len(text) <= 2)
        if scalars["urlish"][token_id] > 0.0 or scalars["fieldish"][token_id] > 0.0:
            token_class[token_id] = 4
        elif scalars["markupish"][token_id] > 0.0:
            token_class[token_id] = 5
        elif scalars["digit_ratio"][token_id] >= 0.5:
            token_class[token_id] = 3
        elif scalars["punct_ratio"][token_id] >= 0.6:
            token_class[token_id] = 6
        elif scalars["upper_ratio"][token_id] >= 0.9 and alpha > 0:
            token_class[token_id] = 2
        elif scalars["titleish"][token_id] > 0.0:
            token_class[token_id] = 7
        elif scalars["alpha_ratio"][token_id] >= 0.5:
            token_class[token_id] = 1
    return {"token_class": torch.from_numpy(token_class), **{name: torch.from_numpy(values) for name, values in scalars.items()}}


def bytes_per_targets(prev_ids: Tensor, target_ids: Tensor, luts: tuple[Tensor, Tensor, Tensor] | None) -> float:
    if luts is None:
        return float("nan")
    base_bytes, has_leading_space, is_boundary = luts
    prev_cpu = prev_ids.detach().cpu().reshape(-1)
    tgt_cpu = target_ids.detach().cpu().reshape(-1)
    token_bytes = base_bytes[tgt_cpu].to(torch.float64)
    token_bytes += (has_leading_space[tgt_cpu] & ~is_boundary[prev_cpu]).to(torch.float64)
    return float(token_bytes.sum().item())


def bpb_from_loss(loss_value: float, token_count: int, byte_count: float) -> float | None:
    if token_count <= 0 or not math.isfinite(byte_count) or byte_count <= 0.0:
        return None
    bits_per_token = loss_value / math.log(2.0)
    return float(bits_per_token * (token_count / byte_count))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos()[None, None, :, :].to(dtype=dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=dtype)
        self._seq_len_cached = seq_len
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            return self._build_cache(seq_len, device, dtype)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


def clamp_quant_bits(bits: int) -> int:
    return int(max(1, min(bits, 8)))


def fake_quantize_weight_ste(weight: Tensor, bits: int) -> Tensor:
    bits = clamp_quant_bits(bits)
    if not weight.is_floating_point():
        return weight
    w32 = weight.float()
    if bits == 1:
        if w32.ndim >= 2:
            scale = w32.flatten(1).abs().mean(dim=1, keepdim=True).clamp_min(1e-8).view(w32.size(0), *([1] * (w32.ndim - 1)))
        else:
            scale = w32.abs().mean().clamp_min(1e-8)
        quantized = torch.where(w32 >= 0, scale, -scale)
    else:
        qmax = max((1 << (bits - 1)) - 1, 1)
        if w32.ndim >= 2:
            scale = w32.flatten(1).abs().amax(dim=1, keepdim=True)
            scale = (scale / qmax).clamp_min(1e-8).view(w32.size(0), *([1] * (w32.ndim - 1)))
        else:
            scale = (w32.abs().max() / qmax).clamp_min(1e-8)
        quantized = torch.clamp(torch.round(w32 / scale), -qmax, qmax) * scale
    return weight + (quantized.to(dtype=weight.dtype) - weight).detach()


def linear_with_optional_qat(x: Tensor, layer: nn.Linear, qat_enabled: bool, qat_bits: int) -> Tensor:
    weight = fake_quantize_weight_ste(layer.weight, qat_bits) if qat_enabled else layer.weight
    return F.linear(x, weight, layer.bias)


def embedding_with_optional_qat(input_ids: Tensor, embedding: nn.Embedding, qat_enabled: bool, qat_bits: int) -> Tensor:
    weight = fake_quantize_weight_ste(embedding.weight, qat_bits) if qat_enabled else embedding.weight
    return F.embedding(input_ids, weight)


INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def torch_dtype_from_name(name: str) -> torch.dtype:
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype name in export: {name}")
    return getattr(torch, name)


def classify_export_tensor(name: str, tensor: Tensor) -> str:
    if not tensor.is_floating_point():
        return "passthrough"
    if tensor.ndim != 2 or tensor.numel() < 4096:
        return "passthrough"
    lname = name.lower()
    if any(key in lname for key in ("wte.weight", "wpe.weight", "lm_head.weight")):
        return "embed"
    if any(key in lname for key in ("ln_", ".ln", "layernorm", "norm.weight")):
        return "passthrough"
    return "matrix"


def pack_lowbit_tensor(values: Tensor, bits: int) -> Tensor:
    bits = clamp_quant_bits(bits)
    flat = values.reshape(-1).to(torch.int64).cpu().tolist()
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    for value in flat:
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits > 0:
        out.append(acc & 0xFF)
    return torch.tensor(list(out), dtype=torch.uint8)


def unpack_lowbit_tensor(packed: Tensor, bits: int, numel: int) -> Tensor:
    bits = clamp_quant_bits(bits)
    raw = packed.cpu().tolist()
    values = np.zeros((numel,), dtype=np.int16)
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1
    idx = 0
    for byte in raw:
        acc |= int(byte) << acc_bits
        acc_bits += 8
        while acc_bits >= bits and idx < numel:
            values[idx] = acc & mask
            acc >>= bits
            acc_bits -= bits
            idx += 1
    return torch.from_numpy(values)


def quantize_tensor_for_export(name: str, tensor: Tensor, bits: int) -> dict[str, Any]:
    bits = clamp_quant_bits(bits)
    t = tensor.detach().cpu().contiguous()
    t32 = t.float()
    rowwise = t32.ndim >= 2
    if bits == 1:
        if rowwise:
            scale = t32.flatten(1).abs().mean(dim=1, keepdim=True).clamp_min(1e-8).view(t32.size(0), *([1] * (t32.ndim - 1)))
        else:
            scale = t32.abs().mean().clamp_min(1e-8)
        packed = pack_lowbit_tensor((t32 >= 0).to(torch.int16), bits)
        return {
            "bits": bits,
            "qmax": 1,
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
            "scale": scale.reshape(-1).to(torch.float16 if rowwise else torch.float32),
            "packed": packed,
            "numel": int(t.numel()),
            "name": name,
            "rowwise": rowwise,
            "binary": True,
        }
    qmax = max((1 << (bits - 1)) - 1, 1)
    if rowwise:
        scale = t32.flatten(1).abs().amax(dim=1, keepdim=True)
        scale = (scale / qmax).clamp_min(1e-8).view(t32.size(0), *([1] * (t32.ndim - 1)))
    else:
        scale = (t32.abs().max() / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round(t32 / scale), -qmax, qmax).to(torch.int16)
    unsigned = (q + qmax).to(torch.int16)
    packed = pack_lowbit_tensor(unsigned, bits)
    return {
        "bits": bits,
        "qmax": qmax,
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "scale": scale.reshape(-1).to(torch.float16 if rowwise else torch.float32),
        "packed": packed,
        "numel": int(t.numel()),
        "name": name,
        "rowwise": rowwise,
    }


def dequantize_tensor_from_export(entry: dict[str, Any]) -> Tensor:
    bits = int(entry["bits"])
    shape = tuple(int(x) for x in entry["shape"])
    numel = int(entry["numel"])
    dtype = torch_dtype_from_name(str(entry["dtype"]))
    unsigned = unpack_lowbit_tensor(entry["packed"], bits, numel)
    scale = entry["scale"].float()
    if bool(entry["rowwise"]):
        scale = scale.view(shape[0], *([1] * (len(shape) - 1)))
    if bool(entry.get("binary", False)):
        signed = unsigned.to(torch.float32).view(shape) * 2.0 - 1.0
        return (signed * scale).to(dtype=dtype)
    qmax = int(entry["qmax"])
    q = unsigned.to(torch.int32) - qmax
    qf = q.to(torch.float32).view(shape)
    return (qf * scale).to(dtype=dtype)


def build_quantized_export_object(state_dict: dict[str, Tensor], plan: dict[str, Any]) -> dict[str, Any]:
    quantized: dict[str, Any] = {}
    passthrough: dict[str, Tensor] = {}
    aliases: dict[str, str] = {}
    if "backbone.wte.weight" in state_dict and "backbone.lm_head.weight" in state_dict and state_dict["backbone.wte.weight"].shape == state_dict["backbone.lm_head.weight"].shape and torch.equal(state_dict["backbone.wte.weight"], state_dict["backbone.lm_head.weight"]):
        aliases["backbone.lm_head.weight"] = "backbone.wte.weight"
    for name, tensor in state_dict.items():
        if name in aliases:
            continue
        category = classify_export_tensor(name, tensor)
        bits = int(plan.get(f"{category}_bits", 0))
        if bits > 0 and category in {"matrix", "embed"}:
            quantized[name] = quantize_tensor_for_export(name, tensor, bits)
            continue
        t = tensor.detach().cpu().contiguous()
        passthrough[name] = t.float() if t.is_floating_point() and t.ndim <= 1 else (t.half() if t.is_floating_point() and t.dtype == torch.float32 else t)
    return {"version": "shadow_export_v1", "plan": plan, "quantized": quantized, "passthrough": passthrough, "aliases": aliases}


def dequantize_export_state_dict(export_obj: dict[str, Any]) -> dict[str, Tensor]:
    state_dict: dict[str, Tensor] = {}
    for name, tensor in export_obj["passthrough"].items():
        state_dict[name] = tensor
    for name, entry in export_obj["quantized"].items():
        state_dict[name] = dequantize_tensor_from_export(entry)
    for name, source in export_obj.get("aliases", {}).items():
        state_dict[name] = state_dict[source]
    return state_dict


def compress_export_object(export_obj: dict[str, Any]) -> bytes:
    raw = io.BytesIO()
    torch.save(export_obj, raw)
    payload = raw.getvalue()
    if _COMPRESSOR == "zstd" and zstandard is not None:
        return zstandard.ZstdCompressor(level=12).compress(payload)
    return zlib.compress(payload, 9)


@torch.no_grad()
def evaluate_loss_only(model: HybridGriffinLanguageModel, val_tokens: Tensor, config: ExperimentConfig, device: torch.device, amp_dtype: torch.dtype, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for batch in iter_validation_batches(val_tokens, config.train.val_batch_tokens, config.model.block_size, max_batches):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            outputs = model(batch["input_ids"], targets=batch["targets"], mode=config.train.mode)
        loss = outputs["loss"] if outputs["loss"] is not None else outputs["gpt_loss"]
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def build_export_candidates(config: ExperimentConfig) -> list[dict[str, Any]]:
    qat_bits = clamp_quant_bits(config.model.qat_bits)
    qat_embed_bits = clamp_quant_bits(config.model.qat_embed_bits if getattr(config.model, "use_selective_qat", False) else max(qat_bits, 6))
    candidates = [
        {"name": "binary_int1_int6", "matrix_bits": 1, "embed_bits": 6},
        {"name": "binary_int1_int8", "matrix_bits": 1, "embed_bits": 8},
        {"name": "ternary_int2_int6", "matrix_bits": 2, "embed_bits": 6},
        {"name": "ternary_int2_int8", "matrix_bits": 2, "embed_bits": 8},
        {"name": "ultra_int3_int6", "matrix_bits": 3, "embed_bits": 6},
        {"name": "aggressive_int4_int6", "matrix_bits": 4, "embed_bits": 6},
        {"name": "balanced_int5_int6", "matrix_bits": 5, "embed_bits": 6},
        {"name": "balanced_int6_int8", "matrix_bits": 6, "embed_bits": 8},
        {"name": "safe_int8_int8", "matrix_bits": 8, "embed_bits": 8},
    ]
    if config.model.qat_enabled:
        candidates.insert(0, {"name": f"qat_aligned_int{qat_bits}_int{qat_embed_bits}", "matrix_bits": qat_bits, "embed_bits": qat_embed_bits})
    return candidates


def evaluate_export_candidates(
    model: HybridGriffinLanguageModel,
    val_tokens: Tensor,
    config: ExperimentConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model = unwrap_model(model)
    original_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    baseline_loss = evaluate_loss_only(model, val_tokens, config, device, amp_dtype, max(1, config.train.export_calibration_batches))
    candidates = build_export_candidates(config)
    best: dict[str, Any] | None = None
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        export_obj = build_quantized_export_object(original_state, candidate)
        compressed = compress_export_object(export_obj)
        model.load_state_dict(dequantize_export_state_dict(export_obj), strict=True)
        candidate_loss = evaluate_loss_only(model, val_tokens, config, device, amp_dtype, max(1, config.train.export_calibration_batches))
        loss_increase = candidate_loss - baseline_loss
        result = {"candidate": candidate, "compressed": compressed, "size_bytes": len(compressed), "loss": candidate_loss, "loss_increase": loss_increase}
        rank0_print(f"export_candidate:{candidate['name']} size_bytes:{result['size_bytes']} loss:{candidate_loss:.4f} delta:{loss_increase:.4f}")
        if loss_increase <= config.train.export_max_loss_increase:
            accepted.append(result)
        if best is None or result["loss_increase"] < best["loss_increase"] or (math.isclose(result["loss_increase"], best["loss_increase"], rel_tol=0.0, abs_tol=1e-6) and result["size_bytes"] < best["size_bytes"]):
            best = result
    chosen = min(accepted, key=lambda item: item["size_bytes"]) if accepted else best
    assert chosen is not None
    model.load_state_dict(original_state, strict=True)
    return baseline_loss, chosen, accepted


def export_quantized_checkpoint(model: HybridGriffinLanguageModel, export_path: str | Path, val_tokens: Tensor, config: ExperimentConfig, device: torch.device, amp_dtype: torch.dtype) -> None:
    baseline_loss, chosen, _ = evaluate_export_candidates(model, val_tokens, config, device, amp_dtype)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(chosen["compressed"])
    rank0_print(
        f"legacy_export_path:{export_path} bytes:{export_path.stat().st_size} "
        f"compressor:{_COMPRESSOR} plan:{chosen['candidate']['name']} "
        f"baseline_loss:{baseline_loss:.4f} export_loss:{chosen['loss']:.4f} "
        f"delta:{chosen['loss_increase']:.4f} threshold:{config.train.export_max_loss_increase:.4f}"
    )


class WebArchaeologyShadow(nn.Module):
    regime_names = ("content", "title", "list", "metadata", "nav", "markup", "numeric", "boilerplate")

    def __init__(self, hidden_size: int, vocab_size: int, config: ShadowConfig, metadata: dict[str, Tensor] | None = None):
        super().__init__()
        self.enabled = config.enabled
        self.window = max(2, int(config.window))
        self.control_scale = float(config.control_scale)
        self.gate_threshold = float(config.gate_threshold)
        self.gate_sharpness = max(float(config.gate_sharpness), 1e-3)
        self.active_target = float(min(max(config.active_target, 0.0), 1.0))
        self.active_excess_coef = float(max(config.active_excess_coef, 0.0))
        self.regime_temp = max(float(config.regime_temp), 1e-3)
        self.boundary_scale = float(config.boundary_scale)
        self.boundary_soft_threshold = float(config.boundary_soft_threshold)
        self.boundary_hard_threshold = max(float(config.boundary_hard_threshold), float(config.boundary_soft_threshold))
        self.suppression_scale = float(config.suppression_scale)
        self.suppression_floor = float(min(max(config.suppression_floor, 0.0), 1.0))
        self.token_dropout_max = float(min(max(config.token_dropout_max, 0.0), 1.0))
        self.corruption_prob = float(min(max(config.corruption_prob, 0.0), 1.0))
        self.fingerprint_buckets = max(8, int(config.fingerprint_buckets))
        self.fingerprint_scale = float(config.fingerprint_scale)
        self.recurrence_scale = float(config.recurrence_scale)
        self.pos_warp_scale = float(config.pos_warp_scale)
        self.adapter_scale = float(config.adapter_scale)
        self.adapter_layers = max(0, int(config.adapter_layers))
        self.reset_scale = float(config.reset_scale)
        self.logit_scale = float(config.logit_scale)
        self.template_slots = max(1, int(config.template_slots))
        self.template_scale = float(config.template_scale)
        self.ancestry_scale = float(config.ancestry_scale)
        self.ln_mod_scale = float(config.ln_mod_scale)
        table_size = max(1, vocab_size)
        meta = metadata or {}
        token_class = meta.get("token_class", torch.zeros((table_size,), dtype=torch.long))
        if token_class.numel() < table_size:
            token_class = F.pad(token_class, (0, table_size - token_class.numel()))
        self.register_buffer("token_class", token_class.long(), persistent=False)
        self.scalar_names = (
            "leading_space",
            "alpha_ratio",
            "digit_ratio",
            "punct_ratio",
            "upper_ratio",
            "titleish",
            "urlish",
            "markupish",
            "fieldish",
            "shortish",
        )
        for name in self.scalar_names:
            value = meta.get(name, torch.zeros((table_size,), dtype=torch.float32))
            if value.numel() < table_size:
                value = F.pad(value, (0, table_size - value.numel()))
            self.register_buffer(name, value.float(), persistent=False)
        self.class_embed = nn.Embedding(8, hidden_size)
        self.regime_embed = nn.Embedding(len(self.regime_names), hidden_size)
        self.boundary_embed = nn.Embedding(3, hidden_size)
        self.fingerprint_embed = nn.Embedding(self.fingerprint_buckets, hidden_size)
        self.recurrence_embed = nn.Embedding(4, hidden_size)
        self.ancestry_embed = nn.Embedding(4, hidden_size)
        dense_feature_dim = len(self.scalar_names) + 8
        self.scalar_proj = nn.Linear(dense_feature_dim, hidden_size)
        self.rhythm_proj = nn.Linear(dense_feature_dim, hidden_size)
        self.multi_view_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.multi_view_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.template_out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + dense_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.bias_net = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.pos_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.adapter_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.logit_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.consistency_head = nn.Linear(hidden_size, len(self.regime_names))
        self.ancestry_head = nn.Linear(hidden_size, 4)
        self.ln_gain_proj = nn.Linear(hidden_size, hidden_size)
        self.ln_bias_proj = nn.Linear(hidden_size, hidden_size)
        self.template_queries = nn.Parameter(torch.randn(self.template_slots, hidden_size) * 0.02)
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(config.gate_bias)

    def _rolling_mean(self, values: Tensor) -> Tensor:
        left_pad = self.window // 2
        right_pad = max(self.window - 1 - left_pad, 0)
        x = values.transpose(1, 2)
        x = F.pad(x, (left_pad, right_pad), mode="replicate")
        return F.avg_pool1d(x, kernel_size=self.window, stride=1).transpose(1, 2)

    def forward(self, input_ids: Tensor, gpt_logits: Tensor, hidden: Tensor) -> dict[str, Tensor]:
        if not self.enabled:
            zeros = torch.zeros_like(hidden)
            regime_probs = torch.zeros((*input_ids.shape, len(self.regime_names)), device=input_ids.device, dtype=torch.float32)
            regime_probs[..., 0] = 1.0
            return {
                "control_bias": zeros,
                "token_scale": torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.float32),
                "pos_bias": zeros,
                "adapter_bias": zeros,
                "logit_bias": torch.zeros((*input_ids.shape, gpt_logits.size(-1)), device=input_ids.device, dtype=torch.float32),
                "ln_gain": zeros,
                "ln_bias": zeros,
                "adapter_norm": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "logit_bias_norm": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "ln_mod_norm": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "gate": torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.float32),
                "active_frac": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "gate_raw_mean": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "bias_norm": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "boundary_rate": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "hard_boundary_rate": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "suppression_mean": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "dropout_rate": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "corruption_rate": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "fingerprint_diversity": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "recurrence_tag_mean": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "template_slot_norm": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "template_usage_entropy": torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.float32),
                "ancestry_logits": torch.zeros((*input_ids.shape, 4), device=input_ids.device, dtype=torch.float32),
                "regime_probs": regime_probs,
                "consistency_logits": torch.zeros((*input_ids.shape, len(self.regime_names)), device=input_ids.device, dtype=torch.float32),
                "augmented_input_ids": input_ids,
                "ancestry_targets": torch.zeros_like(input_ids),
            }
        ids = input_ids.long()
        class_ids = self.token_class[ids]
        scalar_features = torch.stack([getattr(self, name)[ids] for name in self.scalar_names], dim=-1)
        local = self._rolling_mean(scalar_features)
        repeated = torch.zeros_like(ids, dtype=torch.float32)
        repeated[:, 1:] = (ids[:, 1:] == ids[:, :-1]).float()
        repeat_local = self._rolling_mean(repeated.unsqueeze(-1)).squeeze(-1)
        vocab_uniques = torch.zeros_like(repeated)
        for offset in range(1, min(self.window, ids.size(1))):
            vocab_uniques[:, offset:] += (ids[:, offset:] != ids[:, :-offset]).float()
        novelty = (vocab_uniques / max(min(self.window, ids.size(1)) - 1, 1)).clamp(0.0, 1.0)
        probs = F.softmax(gpt_logits.float(), dim=-1)
        log_probs = F.log_softmax(gpt_logits.float(), dim=-1)
        entropy = (-(probs * log_probs).sum(dim=-1) / math.log(probs.size(-1))).clamp(0.0, 1.0)
        top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
        margin = top2[..., 0] - (top2[..., 1] if top2.size(-1) > 1 else 0.0)
        alpha_local = local[..., self.scalar_names.index("alpha_ratio")]
        digit_local = local[..., self.scalar_names.index("digit_ratio")]
        punct_local = local[..., self.scalar_names.index("punct_ratio")]
        upper_local = local[..., self.scalar_names.index("upper_ratio")]
        title_local = local[..., self.scalar_names.index("titleish")]
        url_local = local[..., self.scalar_names.index("urlish")]
        markup_local = local[..., self.scalar_names.index("markupish")]
        field_local = local[..., self.scalar_names.index("fieldish")]
        short_local = local[..., self.scalar_names.index("shortish")]
        space_local = local[..., self.scalar_names.index("leading_space")]
        content = 0.7 * alpha_local + 0.4 * novelty + 0.2 * margin - 0.3 * (url_local + markup_local + digit_local)
        title = 0.9 * title_local + 0.8 * upper_local + 0.3 * space_local - 0.2 * digit_local
        listish = 0.7 * digit_local + 0.6 * punct_local + 0.4 * short_local + 0.3 * space_local
        metadata = 1.0 * url_local + 0.8 * field_local + 0.5 * digit_local + 0.3 * punct_local
        nav = 0.8 * short_local + 0.5 * repeat_local + 0.4 * field_local + 0.3 * url_local - 0.2 * margin
        markup = 1.0 * markup_local + 0.8 * punct_local + 0.4 * url_local
        numeric = 1.1 * digit_local + 0.4 * field_local + 0.3 * punct_local
        boilerplate = 0.9 * repeat_local + 0.5 * short_local + 0.3 * punct_local + 0.2 * url_local - 0.2 * novelty
        regime_logits = torch.stack((content, title, listish, metadata, nav, markup, numeric, boilerplate), dim=-1)
        regime_probs = F.softmax(regime_logits / self.regime_temp, dim=-1)
        regime_shift = torch.zeros_like(repeat_local)
        regime_shift[:, 1:] = (regime_probs[:, 1:, :] - regime_probs[:, :-1, :]).abs().mean(dim=-1)
        boundary_strength = torch.sigmoid(3.0 * regime_shift + 1.4 * title_local + 1.1 * field_local + 0.7 * punct_local + 0.5 * space_local - 0.8 * repeat_local - 0.3 * url_local)
        boundary_ids = torch.where(
            boundary_strength > self.boundary_hard_threshold,
            torch.full_like(class_ids, 2),
            torch.where(boundary_strength > self.boundary_soft_threshold, torch.ones_like(class_ids), torch.zeros_like(class_ids)),
        )
        boundary_emb = self.boundary_embed(boundary_ids)
        recurrence_ids = torch.zeros_like(class_ids)
        recurrence_ids = torch.where(repeat_local > 0.18, torch.ones_like(recurrence_ids), recurrence_ids)
        recurrence_ids = torch.where((repeat_local > 0.28) & (regime_probs[..., self.regime_names.index("boilerplate")] > 0.35), torch.full_like(recurrence_ids, 2), recurrence_ids)
        recurrence_ids = torch.where((boundary_ids == 2) & (repeat_local > 0.12), torch.full_like(recurrence_ids, 3), recurrence_ids)
        fingerprint_ids = (
            class_ids * 13
            + regime_logits.argmax(dim=-1) * 17
            + boundary_ids * 19
            + recurrence_ids * 23
            + (repeat_local * 7.0).long()
        ) % self.fingerprint_buckets
        fingerprint_emb = self.fingerprint_embed(fingerprint_ids)
        dense_features = torch.cat([scalar_features, local[..., :4], repeat_local.unsqueeze(-1), novelty.unsqueeze(-1), entropy.unsqueeze(-1), margin.unsqueeze(-1)], dim=-1)
        lexical_summary = (
            self.class_embed(class_ids)
            + regime_probs @ self.regime_embed.weight
            + self.boundary_scale * boundary_emb
            + self.fingerprint_scale * fingerprint_emb
            + self.recurrence_scale * self.recurrence_embed(recurrence_ids)
        )
        rhythm_summary = self.scalar_proj(dense_features.float()) + self.rhythm_proj(dense_features.float())
        multi_view_input = torch.cat([lexical_summary, rhythm_summary], dim=-1)
        mix_gate = torch.sigmoid(self.multi_view_gate(multi_view_input))
        mixed_summary = mix_gate * lexical_summary + (1.0 - mix_gate) * rhythm_summary
        shadow_summary = torch.tanh(self.multi_view_proj(multi_view_input)) + mixed_summary
        slot_scores = torch.einsum("bth,sh->bts", shadow_summary, self.template_queries) / math.sqrt(shadow_summary.size(-1))
        token_to_slot = F.softmax(slot_scores, dim=-1)
        slot_memory = torch.einsum("bts,bth->bsh", token_to_slot, shadow_summary)
        slot_memory = slot_memory / slot_memory.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_to_token = F.softmax(torch.einsum("bth,bsh->bts", shadow_summary, slot_memory) / math.sqrt(shadow_summary.size(-1)), dim=-1)
        template_context = torch.einsum("bts,bsh->bth", slot_to_token, slot_memory)
        shadow_summary = shadow_summary + self.template_scale * torch.tanh(self.template_out_proj(torch.cat([shadow_summary, template_context], dim=-1)))
        ancestry_targets = torch.zeros_like(class_ids)
        ancestry_targets = torch.where(regime_probs[..., self.regime_names.index("metadata")] + regime_probs[..., self.regime_names.index("numeric")] > 0.55, torch.ones_like(ancestry_targets), ancestry_targets)
        ancestry_targets = torch.where(regime_probs[..., self.regime_names.index("nav")] + regime_probs[..., self.regime_names.index("markup")] + regime_probs[..., self.regime_names.index("list")] > 0.65, torch.full_like(ancestry_targets, 2), ancestry_targets)
        ancestry_targets = torch.where(regime_probs[..., self.regime_names.index("boilerplate")] + repeat_local > 0.65, torch.full_like(ancestry_targets, 3), ancestry_targets)
        ancestry_logits = self.ancestry_head(shadow_summary.float())
        ancestry_probs = F.softmax(ancestry_logits, dim=-1)
        shadow_summary = shadow_summary + self.ancestry_scale * (ancestry_probs @ self.ancestry_embed.weight)
        suppression_strength = torch.sigmoid(
            3.0 * regime_probs[..., self.regime_names.index("boilerplate")]
            + 2.0 * regime_probs[..., self.regime_names.index("nav")]
            + 1.5 * regime_probs[..., self.regime_names.index("metadata")]
            + 0.8 * repeat_local
            - 1.2 * regime_probs[..., self.regime_names.index("content")]
            - 0.7 * regime_probs[..., self.regime_names.index("title")]
            - 1.0
        )
        gate_input = torch.cat([hidden.detach().float(), shadow_summary.float(), dense_features.float()], dim=-1)
        gate_raw = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)
        gate = torch.sigmoid((gate_raw - self.gate_threshold) * self.gate_sharpness)
        reset_indicator = torch.zeros_like(suppression_strength)
        reset_indicator[:, 1:] = (boundary_ids[:, :-1] == 2).float()
        token_scale = (1.0 - self.suppression_scale * suppression_strength).clamp(min=self.suppression_floor, max=1.0)
        token_scale = (token_scale * (1.0 - self.reset_scale * reset_indicator)).clamp(min=self.suppression_floor, max=1.0)
        dropout_mask = torch.zeros_like(token_scale)
        corruption_mask = torch.zeros_like(token_scale)
        augmented_input_ids = input_ids
        if self.training and self.token_dropout_max > 0.0:
            dropout_mask = (torch.rand_like(token_scale) < (self.token_dropout_max * suppression_strength)).float()
            token_scale = (token_scale * (1.0 - dropout_mask)).clamp(min=self.suppression_floor * 0.5, max=1.0)
        if self.training and self.corruption_prob > 0.0 and input_ids.size(0) > 1:
            corruption_mask = (torch.rand_like(token_scale) < (self.corruption_prob * suppression_strength)).float()
            shifted_ids = input_ids.roll(1, dims=0)
            augmented_input_ids = torch.where(corruption_mask.bool(), shifted_ids, input_ids)
        control_bias = self.control_scale * gate.unsqueeze(-1) * self.bias_net(shadow_summary)
        pos_bias = self.pos_warp_scale * boundary_strength.unsqueeze(-1) * self.pos_proj(shadow_summary)
        adapter_bias = self.adapter_scale * (gate + 0.5 * reset_indicator).unsqueeze(-1) * self.adapter_proj(shadow_summary)
        logit_bias = self.logit_scale * gate.unsqueeze(-1) * self.logit_proj(shadow_summary.float())
        consistency_logits = self.consistency_head(shadow_summary.float())
        ln_gain = self.ln_mod_scale * torch.tanh(self.ln_gain_proj(shadow_summary.float()))
        ln_bias = self.ln_mod_scale * torch.tanh(self.ln_bias_proj(shadow_summary.float()))
        slot_usage = token_to_slot.mean(dim=1)
        slot_entropy = -(slot_usage * torch.log(slot_usage.clamp_min(1e-8))).sum(dim=-1) / math.log(max(self.template_slots, 2))
        return {
            "control_bias": control_bias,
            "token_scale": token_scale,
            "pos_bias": pos_bias,
            "adapter_bias": adapter_bias,
            "logit_bias": logit_bias,
            "ln_gain": ln_gain,
            "ln_bias": ln_bias,
            "adapter_norm": adapter_bias.norm(dim=-1).mean(dim=1),
            "logit_bias_norm": logit_bias.norm(dim=-1).mean(dim=1),
            "ln_mod_norm": (ln_gain.norm(dim=-1) + ln_bias.norm(dim=-1)).mean(dim=1),
            "gate": gate,
            "gate_raw_mean": gate_raw.mean(dim=1),
            "active_frac": (gate > 0.25).float().mean(dim=1),
            "bias_norm": control_bias.norm(dim=-1).mean(dim=1),
            "boundary_rate": (boundary_ids > 0).float().mean(dim=1),
            "hard_boundary_rate": (boundary_ids == 2).float().mean(dim=1),
            "suppression_mean": suppression_strength.mean(dim=1),
            "dropout_rate": dropout_mask.mean(dim=1),
            "corruption_rate": corruption_mask.mean(dim=1),
            "fingerprint_diversity": fingerprint_ids.float().std(dim=1) / max(float(self.fingerprint_buckets), 1.0),
            "recurrence_tag_mean": recurrence_ids.float().mean(dim=1),
            "template_slot_norm": slot_memory.norm(dim=-1).mean(dim=-1),
            "template_usage_entropy": slot_entropy,
            "ancestry_logits": ancestry_logits,
            "ancestry_targets": ancestry_targets,
            "regime_probs": regime_probs,
            "consistency_logits": consistency_logits,
            "augmented_input_ids": augmented_input_ids,
        }


@dataclass
class GriffinModelConfig:
    backbone_type: str = "griffin_recurrent_gemma"
    vocab_size: int = 1024
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    use_flash: bool = True
    local_attn_backend: str = "auto"
    qat_enabled: bool = False
    qat_bits: int = 6
    qat_embed_bits: int = 6
    qat_sensitive_bits: int = 6
    use_selective_qat: bool = True
    local_window: int = 64
    mlp_mult: float = 4.0
    recurrent_variant: str = "rglru"
    recurrent_decay_bias: float = 1.0
    recurrent_slow_decay_bias: float = 2.5
    recurrent_multiscale: bool = True
    recurrent_chunk_size: int = 128
    use_recurrent_cache: bool = True
    use_attention_cache: bool = True
    mix_strategy: str = "learned"
    mix_temperature: float = 1.0
    recurrent_branch_init: float = 1.0
    attention_branch_init: float = 0.5
    use_shadow_recurrent_injection: bool = False


@dataclass
class GriffinLayerCache:
    recurrent_state: Tensor | None = None
    attn_k: Tensor | None = None
    attn_v: Tensor | None = None


@dataclass
class GriffinCache:
    position: int = 0
    layers: list[GriffinLayerCache] = field(default_factory=list)


def flash_attn_available() -> bool:
    return _flash_attn_func is not None


def diagonal_linear_scan(decay: Tensor, inputs: Tensor, initial_state: Tensor | None, chunk_size: int) -> tuple[Tensor, Tensor]:
    if decay.shape != inputs.shape:
        raise ValueError(f"decay and inputs must have the same shape, got {decay.shape} vs {inputs.shape}")
    bsz, seq_len, width = inputs.shape
    state = torch.zeros((bsz, width), device=inputs.device, dtype=torch.float32) if initial_state is None else initial_state.float()
    outputs = torch.empty((bsz, seq_len, width), device=inputs.device, dtype=torch.float32)
    chunk = seq_len if chunk_size <= 0 else min(chunk_size, seq_len)

    def scan_chunk_sequential(a_chunk: Tensor, b_chunk: Tensor, chunk_state: Tensor) -> tuple[Tensor, Tensor]:
        chunk_states = torch.empty_like(b_chunk)
        running = chunk_state
        for offset in range(a_chunk.size(1)):
            running = a_chunk[:, offset, :] * running + b_chunk[:, offset, :]
            chunk_states[:, offset, :] = running
        return chunk_states, running

    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        a = decay[:, start:end, :].float().clamp(min=1e-6, max=1.0 - 1e-6)
        b = inputs[:, start:end, :].float()
        prefix = torch.cumprod(a, dim=1)
        prefix_safe = prefix.clamp_min(1e-20)
        normalized_inputs = b / prefix_safe
        chunk_states = prefix * (torch.cumsum(normalized_inputs, dim=1) + state.unsqueeze(1))
        unstable = (
            (prefix_safe.amin() < 1e-10)
            or (normalized_inputs.abs().amax() > 1e4)
            or (chunk_states.abs().amax() > 1e4)
            or (not torch.isfinite(chunk_states).all())
        )
        if unstable:
            chunk_states, state = scan_chunk_sequential(a, b, state)
        else:
            state = chunk_states[:, -1, :]
        outputs[:, start:end, :] = chunk_states
    return outputs.to(dtype=inputs.dtype), state


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class GriffinLocalAttention(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.window = max(2, min(config.local_window, config.block_size))
        self.use_flash = config.use_flash
        self.backend = config.local_attn_backend
        self.use_attention_cache = config.use_attention_cache
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base) if config.use_rope else None
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self._reported_backend: str | None = None

    def _position_cos_sin(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self.rope is None:
            raise RuntimeError("RoPE is disabled for this module.")
        max_position = int(position_ids.max().item()) + 1
        cos, sin = self.rope(max_position, position_ids.device, dtype)
        return cos[:, :, position_ids, :], sin[:, :, position_ids, :]

    def _apply_rope_positions(self, q: Tensor, k: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if self.rope is None:
            return q, k
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        cos, sin = self._position_cos_sin(position_ids, q_t.dtype)
        q_t = apply_rope(q_t, cos, sin)
        k_t = apply_rope(k_t, cos, sin)
        return q_t.transpose(1, 2), k_t.transpose(1, 2)

    def _local_mask(self, q_len: int, k_len: int, device: torch.device, q_offset: int, k_offset: int) -> Tensor:
        q_positions = q_offset + torch.arange(q_len, device=device)
        k_positions = k_offset + torch.arange(k_len, device=device)
        allowed = (k_positions[None, :] <= q_positions[:, None]) & ((q_positions[:, None] - k_positions[None, :]) < self.window)
        mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=torch.float32)
        return mask.masked_fill(allowed, 0.0)

    def _manual_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(q_t.size(-1))
        att = (q_t @ k_t.transpose(-2, -1)) * scale
        att = att + mask[None, None, :, :]
        att = F.softmax(att.float(), dim=-1).to(dtype=q_t.dtype)
        att = self.attn_dropout(att)
        return (att @ v_t).transpose(1, 2)

    def _sdpa_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask[None, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return y.transpose(1, 2)

    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor | None:
        if _flash_attn_func is None or not self.use_flash:
            return None
        try:
            return _flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window - 1, 0),
            )
        except TypeError:
            try:
                return _flash_attn_func(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    self.dropout if self.training else 0.0,
                    causal=True,
                    window_size=(self.window - 1, 0),
                )
            except Exception:
                return None
        except Exception:
            return None

    def _select_backend(self, device: torch.device, use_cache: bool) -> str:
        if self.backend == "sdpa":
            return "sdpa"
        if self.backend == "flash":
            return "flash" if device.type == "cuda" and flash_attn_available() and not use_cache else "sdpa"
        if self.use_flash and device.type == "cuda" and flash_attn_available() and not use_cache:
            return "flash"
        return "sdpa"

    def forward(
        self,
        x: Tensor,
        layer_cache: GriffinLayerCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[Tensor, GriffinLayerCache | None]:
        bsz, seq_len, _ = x.shape
        qat_on = self.training and self.qat_enabled
        q = linear_with_optional_qat(x, self.q_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim)
        k = linear_with_optional_qat(x, self.k_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim)
        v = linear_with_optional_qat(x, self.v_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim)
        position_ids = position_offset + torch.arange(seq_len, device=x.device, dtype=torch.long)
        q, k = self._apply_rope_positions(q, k, position_ids)

        past_k = layer_cache.attn_k if layer_cache is not None else None
        past_v = layer_cache.attn_v if layer_cache is not None else None
        past_len = 0 if past_k is None else int(past_k.size(1))
        k_full = torch.cat([past_k, k], dim=1) if past_k is not None else k
        v_full = torch.cat([past_v, v], dim=1) if past_v is not None else v
        if k_full.size(1) > self.window:
            k_full = k_full[:, -self.window :, :, :]
            v_full = v_full[:, -self.window :, :, :]
        k_offset = position_offset + seq_len - int(k_full.size(1))
        backend = self._select_backend(x.device, use_cache and self.use_attention_cache)
        if self._reported_backend != backend:
            print(f"local_attention_backend:{backend} window:{self.window} flash_available:{flash_attn_available()}")
            self._reported_backend = backend

        y: Tensor | None = None
        if backend == "flash" and past_len == 0:
            y = self._flash_attention(q, k_full, v_full)
            if y is None or not torch.isfinite(y).all():
                backend = "sdpa"
                y = None
        if backend == "sdpa":
            mask = self._local_mask(seq_len, int(k_full.size(1)), x.device, position_offset, k_offset)
            if supports_fast_attention(x.device, True):
                y = self._sdpa_attention(q, k_full, v_full, mask)
                if not torch.isfinite(y).all():
                    y = self._manual_attention(q, k_full, v_full, mask)
            else:
                y = self._manual_attention(q, k_full, v_full, mask)
        assert y is not None
        if not torch.isfinite(y).all():
            raise FloatingPointError("Non-finite values detected in GriffinLocalAttention output.")
        y = y.contiguous().view(bsz, seq_len, self.n_embd)
        next_cache = None
        if use_cache and self.use_attention_cache:
            next_cache = GriffinLayerCache(attn_k=k_full.detach(), attn_v=v_full.detach())
        return self.resid_dropout(linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits)), next_cache


class GriffinRecurrentPath(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.variant = config.recurrent_variant
        self.chunk_size = max(0, int(config.recurrent_chunk_size))
        self.use_recurrent_cache = config.use_recurrent_cache
        self.in_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.a_param = nn.Parameter(torch.full((config.n_embd,), float(config.recurrent_decay_bias)))
        self.skip_scale = nn.Parameter(torch.ones(config.n_embd))
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)

    def forward(
        self,
        x: Tensor,
        prev_state: Tensor | None = None,
        recurrent_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        qat_on = self.training and self.qat_enabled
        projected = linear_with_optional_qat(x, self.in_proj, qat_on, self.qat_bits)
        candidate, input_gate_logits, output_gate_logits, decay_delta = projected.chunk(4, dim=-1)
        if recurrent_bias is not None:
            candidate = candidate + recurrent_bias
        input_gate = torch.sigmoid(input_gate_logits.float())
        output_gate = torch.sigmoid(output_gate_logits.float())
        log_a = -F.softplus(self.a_param + decay_delta.float())
        decay = torch.exp(log_a).clamp(min=1e-6, max=1.0 - 1e-6)
        candidate_f = candidate.float()
        if self.variant == "rglru_relaxed":
            candidate_value = F.silu(candidate_f)
            output_mix = 1.0
            input_mix = input_gate
            input_scale = 1.0 - decay
        elif self.variant == "legacy_like":
            candidate_value = F.silu(candidate_f)
            output_mix = output_gate
            input_mix = 1.0 - decay
            input_scale = 1.0
        else:
            candidate_value = torch.tanh(candidate_f)
            output_mix = output_gate
            input_mix = input_gate
            input_scale = torch.sqrt((1.0 - decay.square()).clamp_min(1e-6))
        inputs = input_scale * input_mix * candidate_value
        states, final_state = diagonal_linear_scan(decay, inputs, prev_state, self.chunk_size)
        if not torch.isfinite(states).all():
            raise FloatingPointError("Non-finite values detected in GriffinRecurrentPath states.")
        skip = torch.tanh(self.skip_scale).view(1, 1, -1) * input_mix * candidate_value
        y = (output_mix * states + skip).to(dtype=x.dtype)
        y = linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits)
        if not torch.isfinite(y).all():
            raise FloatingPointError("Non-finite values detected in GriffinRecurrentPath output.")
        next_state = final_state.detach() if self.use_recurrent_cache else None
        return self.resid_dropout(y), next_state


class GriffinMLP(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        hidden = max(config.n_embd, int(round(config.n_embd * config.mlp_mult)))
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)

    def forward(self, x: Tensor) -> Tensor:
        qat_on = self.training and self.qat_enabled
        gate = F.silu(linear_with_optional_qat(x, self.gate_proj, qat_on, self.qat_bits))
        value = linear_with_optional_qat(x, self.up_proj, qat_on, self.qat_bits)
        out = linear_with_optional_qat(gate * value, self.down_proj, qat_on, self.qat_bits)
        return self.dropout(out)


class GriffinBlock(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.mix_strategy = config.mix_strategy
        self.use_recurrent_cache = config.use_recurrent_cache
        self.use_attention_cache = config.use_attention_cache
        self.use_shadow_recurrent_injection = config.use_shadow_recurrent_injection
        self.temporal_norm = RMSNorm(config.n_embd)
        self.local_attn = GriffinLocalAttention(config)
        self.recurrent = GriffinRecurrentPath(config)
        self.temporal_mix = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.channel_norm = RMSNorm(config.n_embd)
        self.channel_mlp = GriffinMLP(config)
        nn.init.zeros_(self.temporal_mix.weight)
        nn.init.zeros_(self.temporal_mix.bias)

    def _mix(self, h: Tensor, recurrent_out: Tensor, attn_out: Tensor) -> Tensor:
        if self.mix_strategy == "recurrent_only":
            return recurrent_out
        if self.mix_strategy == "attention_only":
            return attn_out
        if self.mix_strategy == "sum":
            return recurrent_out + attn_out
        mix = torch.sigmoid(self.temporal_mix(h))
        return mix * attn_out + (1.0 - mix) * recurrent_out

    def forward(
        self,
        x: Tensor,
        layer_cache: GriffinLayerCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
        recurrent_injection: Tensor | None = None,
    ) -> tuple[Tensor, GriffinLayerCache | None]:
        h = self.temporal_norm(x)
        prev_state = None if layer_cache is None else layer_cache.recurrent_state
        recurrent_bias = recurrent_injection if self.use_shadow_recurrent_injection else None
        recurrent_out, next_recurrent_state = self.recurrent(h, prev_state=prev_state, recurrent_bias=recurrent_bias)
        attn_out, next_attn_cache = self.local_attn(h, layer_cache=layer_cache, use_cache=use_cache, position_offset=position_offset)
        temporal_out = self._mix(h, recurrent_out, attn_out)
        x = x + self.temporal_dropout(temporal_out)
        x = x + self.channel_mlp(self.channel_norm(x))
        next_cache = None
        if use_cache:
            next_cache = GriffinLayerCache(
                recurrent_state=next_recurrent_state if self.use_recurrent_cache else None,
                attn_k=None if next_attn_cache is None or not self.use_attention_cache else next_attn_cache.attn_k,
                attn_v=None if next_attn_cache is None or not self.use_attention_cache else next_attn_cache.attn_v,
            )
        return x, next_cache


class GriffinBackbone(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.config = config
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.use_recurrent_cache = config.use_recurrent_cache
        self.use_attention_cache = config.use_attention_cache
        self.local_attn_backend = config.local_attn_backend
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GriffinBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_qat(self, enabled: bool, bits: int | None = None) -> None:
        self.qat_enabled = enabled
        if bits is not None:
            self.qat_bits = clamp_quant_bits(bits)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "qat_enabled"):
                module.qat_enabled = enabled
            if bits is not None and hasattr(module, "qat_bits"):
                module.qat_bits = self.qat_bits

    def init_cache(self, batch_size: int) -> GriffinCache:
        return GriffinCache(position=0, layers=[GriffinLayerCache() for _ in range(self.config.n_layer)])

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        return_hidden: bool = True,
        control_bias: Tensor | None = None,
        token_scale: Tensor | None = None,
        pos_bias: Tensor | None = None,
        adapter_bias: Tensor | None = None,
        adapter_layers: int = 0,
        ln_gain: Tensor | None = None,
        ln_bias: Tensor | None = None,
        cache: GriffinCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        qat_on = self.training and self.qat_enabled
        if use_cache and cache is None:
            cache = self.init_cache(bsz)
        position_offset = 0 if cache is None else int(cache.position)
        tok = embedding_with_optional_qat(input_ids, self.wte, qat_on, max(self.qat_bits, 6))
        if self.wpe is not None:
            pos = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device, dtype=torch.long)
            if int(pos[-1].item()) >= self.config.block_size:
                raise ValueError(f"Absolute position {int(pos[-1].item())} exceeds learned positional table size {self.config.block_size}")
            pos_emb = embedding_with_optional_qat(pos, self.wpe, qat_on, max(self.qat_bits, 6))
            x = tok + pos_emb[None, :, :]
        else:
            x = tok
        if pos_bias is not None:
            x = x + pos_bias
        if control_bias is not None:
            x = x + control_bias
        if token_scale is not None:
            x = x * token_scale.unsqueeze(-1)
        x = self.drop(x)
        apply_adapter_from = max(len(self.blocks) - max(adapter_layers, 0), 0)
        next_layers: list[GriffinLayerCache] = []
        recurrent_injection = control_bias if self.config.use_shadow_recurrent_injection else None
        for block_idx, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache.layers[block_idx]
            x, next_layer_cache = block(
                x,
                layer_cache=layer_cache,
                use_cache=use_cache,
                position_offset=position_offset,
                recurrent_injection=recurrent_injection,
            )
            if adapter_bias is not None and block_idx >= apply_adapter_from:
                x = x + adapter_bias
            if use_cache:
                next_layers.append(next_layer_cache or GriffinLayerCache())
        hidden = self.ln_f(x)
        if ln_gain is not None:
            hidden = hidden * (1.0 + ln_gain)
        if ln_bias is not None:
            hidden = hidden + ln_bias
        logits = linear_with_optional_qat(hidden, self.lm_head, qat_on, max(self.qat_bits, 6))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits detected in GriffinBackbone forward pass.")
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss detected in GriffinBackbone forward pass.")
        next_cache = GriffinCache(position=position_offset + seq_len, layers=next_layers) if use_cache else None
        return {"logits": logits, "loss": loss, "hidden_states": hidden if return_hidden else None, "cache": next_cache}

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        out = input_ids
        idx = out[:, -self.config.block_size :]
        step_out = self(idx, return_hidden=False, cache=None, use_cache=True)
        cache = step_out["cache"]
        logits = step_out["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
            step_out = self(next_token, return_hidden=False, cache=cache, use_cache=True)
            cache = step_out["cache"]
            logits = step_out["logits"][:, -1, :]
        return out


class HybridGriffinLanguageModel(nn.Module):
    def __init__(self, model_config: GriffinModelConfig, shadow_config: ShadowConfig, shadow_metadata: dict[str, Tensor] | None = None):
        super().__init__()
        self.backbone = GriffinBackbone(model_config)
        self.model_config = model_config
        self.shadow_config = shadow_config
        self.shadow_stream = WebArchaeologyShadow(model_config.n_embd, model_config.vocab_size, shadow_config, shadow_metadata)

    def forward_backbone(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        cache: GriffinCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        out = self.backbone(input_ids, targets=targets, return_hidden=True, cache=cache, use_cache=use_cache)
        return {
            "logits": out["logits"],
            "loss": out["loss"],
            "hidden_states": out["hidden_states"],
            "gpt_logits": out["logits"],
            "gpt_loss": out["loss"],
            "cache": out.get("cache"),
        }

    def forward_shadow(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        with torch.no_grad():
            base_out = self.backbone(input_ids, targets=None, return_hidden=True)
            gpt_logits = base_out["logits"]
            hidden = base_out["hidden_states"]
        assert hidden is not None
        shadow_out = self.shadow_stream(input_ids, gpt_logits.detach(), hidden.detach())
        controlled_out = self.backbone(
            shadow_out["augmented_input_ids"],
            targets=None,
            return_hidden=True,
            control_bias=shadow_out["control_bias"],
            token_scale=shadow_out["token_scale"],
            pos_bias=shadow_out["pos_bias"],
            adapter_bias=shadow_out["adapter_bias"],
            adapter_layers=self.shadow_config.adapter_layers,
            ln_gain=shadow_out["ln_gain"],
            ln_bias=shadow_out["ln_bias"],
        )
        logits = controlled_out["logits"] + shadow_out["logit_bias"]
        outputs: dict[str, Tensor | None] = {
            "logits": logits,
            "loss": None,
            "gpt_logits": gpt_logits,
            "gpt_loss": None,
            "hidden_states": controlled_out["hidden_states"],
            "shadow_gate": shadow_out["gate"],
            "shadow_gate_raw_mean": shadow_out["gate_raw_mean"],
            "shadow_active_frac": shadow_out["active_frac"],
            "shadow_bias_norm": shadow_out["bias_norm"],
            "shadow_boundary_rate": shadow_out["boundary_rate"],
            "shadow_hard_boundary_rate": shadow_out["hard_boundary_rate"],
            "shadow_suppression_mean": shadow_out["suppression_mean"],
            "shadow_token_scale_mean": shadow_out["token_scale"].mean(dim=1),
            "shadow_dropout_rate": shadow_out["dropout_rate"],
            "shadow_corruption_rate": shadow_out["corruption_rate"],
            "shadow_fingerprint_diversity": shadow_out["fingerprint_diversity"],
            "shadow_recurrence_tag_mean": shadow_out["recurrence_tag_mean"],
            "shadow_template_slot_norm": shadow_out["template_slot_norm"],
            "shadow_template_usage_entropy": shadow_out["template_usage_entropy"],
            "shadow_adapter_norm": shadow_out["adapter_norm"],
            "shadow_logit_bias_norm": shadow_out["logit_bias_norm"],
            "shadow_ln_mod_norm": shadow_out["ln_mod_norm"],
            "shadow_regime_probs": shadow_out["regime_probs"],
            "shadow_consistency_logits": shadow_out["consistency_logits"],
            "shadow_ancestry_logits": shadow_out["ancestry_logits"],
        }
        if targets is not None:
            outputs["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["gpt_loss"] = F.cross_entropy(gpt_logits.reshape(-1, gpt_logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["shadow_target_logprobs"] = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["gpt_target_logprobs"] = F.log_softmax(gpt_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["shadow_consistency_loss"] = F.cross_entropy(
                shadow_out["consistency_logits"][:, :-1, :].reshape(-1, shadow_out["consistency_logits"].size(-1)).float(),
                shadow_out["regime_probs"][:, 1:, :].argmax(dim=-1).reshape(-1),
                reduction="mean",
            )
            outputs["shadow_ancestry_loss"] = F.cross_entropy(
                shadow_out["ancestry_logits"].reshape(-1, shadow_out["ancestry_logits"].size(-1)).float(),
                shadow_out["ancestry_targets"].reshape(-1),
                reduction="mean",
            )
        return outputs

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        mode: str = "backbone",
        cache: GriffinCache | None = None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        if mode == "shadow":
            return self.forward_shadow(input_ids, targets=targets)
        return self.forward_backbone(input_ids, targets=targets, cache=cache, use_cache=use_cache)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_model_for_mode(model: HybridGriffinLanguageModel, mode: str) -> None:
    model = unwrap_model(model)
    if mode == "shadow":
        set_requires_grad(model.backbone, False)
        set_requires_grad(model.shadow_stream, True)
    else:
        set_requires_grad(model.backbone, True)
        set_requires_grad(model.shadow_stream, False)


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    # Fast orthogonalization used by Muon on matrix-shaped updates.
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.to(dtype=torch.bfloat16)
    x = x / (x.norm() + eps)
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.transpose(0, 1)
    for _ in range(steps):
        a_mat = x @ x.transpose(0, 1)
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return x.transpose(0, 1) if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params: list[Tensor], lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.ndim != 2:
                    raise ValueError(f"Muon expects only 2D parameters, got shape {tuple(grad.shape)}")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                update = update * max(1.0, float(update.size(0)) / float(update.size(1))) ** 0.5
                param.add_(update.to(dtype=param.dtype), alpha=-lr)
        return loss


class OptimizerBundle:
    def __init__(self, optimizers: list[tuple[str, torch.optim.Optimizer]]):
        self.optimizers = [(name, opt) for name, opt in optimizers if opt.param_groups]

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for _, optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for _, optimizer in self.optimizers:
            merged.update(optimizer.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for _, optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "optimizer_bundle_v1", "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizers = state_dict.get("optimizers")
        if not isinstance(optimizers, dict):
            raise ValueError("Optimizer state is not an optimizer bundle")
        for name, optimizer in self.optimizers:
            if name not in optimizers:
                raise ValueError(f"Missing optimizer state for {name}")
            optimizer.load_state_dict(optimizers[name])


def split_legacy_backbone_params(backbone: nn.Module) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    muon_params: list[Tensor] = []
    embed_head_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    seen: set[int] = set()
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim == 2 and not (
            name == "wte.weight" or name == "lm_head.weight" or name == "wpe.weight"
        ):
            muon_params.append(param)
        elif param.ndim >= 2:
            embed_head_params.append(param)
        else:
            scalar_params.append(param)
    return muon_params, embed_head_params, scalar_params


def build_optimizer(model: HybridGriffinLanguageModel, config: TrainConfig) -> torch.optim.Optimizer | OptimizerBundle:
    model = unwrap_model(model)
    if config.mode == "shadow":
        params = [p for p in model.shadow_stream.parameters() if p.requires_grad]
        lr = config.shadow_lr
        optimizer_type = config.shadow_optimizer
    else:
        params = [p for p in model.backbone.parameters() if p.requires_grad]
        lr = config.backbone_lr
        optimizer_type = config.backbone_optimizer
    if not params:
        raise RuntimeError(f"No trainable parameters found for mode={config.mode}")
    if config.mode == "shadow" or optimizer_type == "adamw":
        decay_params = [p for p in params if p.ndim >= 2]
        no_decay_params = [p for p in params if p.ndim < 2]
        groups: list[dict[str, Any]] = []
        if decay_params:
            groups.append({"params": decay_params, "lr": lr, "base_lr": lr, "weight_decay": config.weight_decay})
        if no_decay_params:
            groups.append({"params": no_decay_params, "lr": lr, "base_lr": lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, foreach=False)
    if config.mode != "backbone":
        raise ValueError(f"Unsupported optimizer {optimizer_type} for mode={config.mode}")
    if optimizer_type != "muon_adamw":
        raise ValueError(f"Unsupported backbone optimizer: {optimizer_type}")

    muon_params, embed_head_params, scalar_params = split_legacy_backbone_params(model.backbone)
    optimizers: list[tuple[str, torch.optim.Optimizer]] = []
    if muon_params:
        muon_optimizer = Muon(
            muon_params,
            lr=lr,
            momentum=config.muon_momentum,
            backend_steps=config.muon_backend_steps,
            nesterov=config.muon_nesterov,
        )
        for group in muon_optimizer.param_groups:
            group["base_lr"] = lr
        optimizers.append(("muon", muon_optimizer))
    if embed_head_params:
        embed_lr = lr * config.backbone_embed_lr_scale
        embed_optimizer = torch.optim.AdamW(
            [{"params": embed_head_params, "lr": embed_lr, "base_lr": embed_lr, "weight_decay": config.weight_decay}],
            foreach=False,
        )
        optimizers.append(("adamw_embed", embed_optimizer))
    if scalar_params:
        scalar_lr = lr * config.backbone_scalar_lr_scale
        scalar_optimizer = torch.optim.AdamW(
            [{"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr, "weight_decay": 0.0}],
            foreach=False,
        )
        optimizers.append(("adamw_scalar", scalar_optimizer))
    if not optimizers:
        raise RuntimeError("No optimizer groups were created for muon_adamw")
    rank0_print(
        "optimizer_setup "
        f"mode:{config.mode} optimizer:{optimizer_type} "
        f"muon_tensors:{len(muon_params)} muon_params:{sum(p.numel() for p in muon_params)} "
        f"embed_tensors:{len(embed_head_params)} embed_params:{sum(p.numel() for p in embed_head_params)} "
        f"scalar_tensors:{len(scalar_params)} scalar_params:{sum(p.numel() for p in scalar_params)} "
        f"muon_lr:{lr:.6g} embed_lr:{lr * config.backbone_embed_lr_scale:.6g} "
        f"scalar_lr:{lr * config.backbone_scalar_lr_scale:.6g}"
    )
    return OptimizerBundle(optimizers)


def lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min((step + 1) / warmup_steps, 1.0)


def apply_lr_schedule(optimizer: torch.optim.Optimizer | OptimizerBundle, base_lr: float, scale: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", base_lr) * scale


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer | OptimizerBundle, device: torch.device) -> None:
    if isinstance(optimizer, OptimizerBundle):
        for _, nested in optimizer.optimizers:
            move_optimizer_state_to_device(nested, device)
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_update_muon_momentum(optimizer: torch.optim.Optimizer | OptimizerBundle, config: TrainConfig, step: int) -> None:
    if not isinstance(optimizer, OptimizerBundle):
        return
    if config.backbone_optimizer != "muon_adamw":
        return
    if config.muon_momentum_warmup_steps > 0:
        frac = min(float(step) / float(config.muon_momentum_warmup_steps), 1.0)
    else:
        frac = 1.0
    momentum = (1.0 - frac) * config.muon_momentum_warmup_start + frac * config.muon_momentum
    for name, nested in optimizer.optimizers:
        if name != "muon":
            continue
        for group in nested.param_groups:
            group["momentum"] = momentum


def log_train_step(step: int, loss: float, extra: dict[str, float]) -> None:
    pieces = [f"step:{step}", f"loss:{loss:.4f}"] + [f"{k}:{v:.4f}" for k, v in extra.items()]
    rank0_print(" ".join(pieces))


def build_metrics_row(kind: str, step: int, mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": kind, "step": step, "mode": mode}
    for key, value in sorted(metrics.items()):
        row[key] = value
    return row


def compute_shadow_losses(outputs: dict[str, Tensor], config: ShadowConfig) -> tuple[Tensor, dict[str, float]]:
    ce_loss = outputs["loss"]
    assert ce_loss is not None and outputs["gpt_loss"] is not None
    gate_mean = outputs["shadow_gate"].mean()
    suppression_mean = outputs["shadow_suppression_mean"].mean()
    consistency_loss = outputs["shadow_consistency_loss"]
    ancestry_loss = outputs["shadow_ancestry_loss"]
    active_frac = outputs["shadow_active_frac"].mean()
    active_excess = torch.relu(active_frac - config.active_target)
    gate_loss = config.gate_l1_coef * gate_mean + config.suppression_l1_coef * suppression_mean + config.active_excess_coef * active_excess
    template_norm = outputs["shadow_template_slot_norm"].mean()
    adapter_norm = outputs["shadow_adapter_norm"].mean()
    logit_bias_norm = outputs["shadow_logit_bias_norm"].mean()
    ln_mod_norm = outputs["shadow_ln_mod_norm"].mean()
    ancestry_probs = F.softmax(outputs["shadow_ancestry_logits"].float(), dim=-1)
    ancestry_mean = ancestry_probs.mean(dim=(0, 1))
    ancestry_entropy = -(ancestry_mean * torch.log(ancestry_mean.clamp_min(1e-8))).sum() / math.log(ancestry_mean.numel())
    slot_balance_penalty = 1.0 - outputs["shadow_template_usage_entropy"].mean()
    regularization = (
        config.template_norm_coef * template_norm
        + config.adapter_norm_coef * adapter_norm
        + config.logit_norm_coef * logit_bias_norm
        + config.ln_norm_coef * ln_mod_norm
        + config.slot_balance_coef * slot_balance_penalty
        + config.ancestry_entropy_coef * (1.0 - ancestry_entropy)
    )
    total_loss = ce_loss + gate_loss + config.consistency_coef * consistency_loss + config.ancestry_coef * ancestry_loss + regularization
    stats = {
        "ce_loss": float(ce_loss.item()),
        "gpt_loss": float(outputs["gpt_loss"].item()),
        "shadow_gate_mean": float(gate_mean.item()),
        "shadow_gate_raw_mean": float(outputs["shadow_gate_raw_mean"].mean().item()),
        "shadow_active_frac": float(active_frac.item()),
        "shadow_active_excess": float(active_excess.item()),
        "shadow_bias_norm": float(outputs["shadow_bias_norm"].mean().item()),
        "shadow_boundary_rate": float(outputs["shadow_boundary_rate"].mean().item()),
        "shadow_hard_boundary_rate": float(outputs["shadow_hard_boundary_rate"].mean().item()),
        "shadow_suppression_mean": float(suppression_mean.item()),
        "shadow_token_scale_mean": float(outputs["shadow_token_scale_mean"].mean().item()),
        "shadow_dropout_rate": float(outputs["shadow_dropout_rate"].mean().item()),
        "shadow_corruption_rate": float(outputs["shadow_corruption_rate"].mean().item()),
        "shadow_fingerprint_diversity": float(outputs["shadow_fingerprint_diversity"].mean().item()),
        "shadow_recurrence_tag_mean": float(outputs["shadow_recurrence_tag_mean"].mean().item()),
        "shadow_template_slot_norm": float(outputs["shadow_template_slot_norm"].mean().item()),
        "shadow_template_usage_entropy": float(outputs["shadow_template_usage_entropy"].mean().item()),
        "shadow_adapter_norm": float(adapter_norm.item()),
        "shadow_logit_bias_norm": float(logit_bias_norm.item()),
        "shadow_ln_mod_norm": float(ln_mod_norm.item()),
        "shadow_consistency_loss": float(consistency_loss.item()),
        "shadow_ancestry_loss": float(ancestry_loss.item()),
        "shadow_ancestry_entropy": float(ancestry_entropy.item()),
        "shadow_slot_balance_penalty": float(slot_balance_penalty.item()),
        "shadow_target_lp": float(outputs["shadow_target_logprobs"].mean().item()),
        "gpt_target_lp": float(outputs["gpt_target_logprobs"].mean().item()),
    }
    regime_mean = outputs["shadow_regime_probs"].mean(dim=(0, 1))
    for idx, name in enumerate(WebArchaeologyShadow.regime_names):
        stats[f"shadow_regime_{name}"] = float(regime_mean[idx].item())
    for idx, name in enumerate(("content", "metadata", "layout", "template")):
        stats[f"shadow_ancestry_{name}"] = float(ancestry_mean[idx].item())
    return total_loss, stats


@torch.no_grad()
def evaluate(model: HybridGriffinLanguageModel, val_tokens: Tensor, config: ExperimentConfig, device: torch.device, amp_dtype: torch.dtype, tokenizer_luts: tuple[Tensor, Tensor, Tensor] | None = None) -> dict[str, float]:
    model.eval()
    shadow_mode = config.train.mode == "shadow"
    losses: list[float] = []
    gpt_losses: list[float] = []
    shadow_gate_means: list[float] = []
    shadow_gate_raw_means: list[float] = []
    shadow_active_fracs: list[float] = []
    shadow_bias_norms: list[float] = []
    shadow_boundary_rates: list[float] = []
    shadow_hard_boundary_rates: list[float] = []
    shadow_suppression_means: list[float] = []
    shadow_token_scale_means: list[float] = []
    shadow_dropout_rates: list[float] = []
    shadow_corruption_rates: list[float] = []
    shadow_fingerprint_diversities: list[float] = []
    shadow_recurrence_tag_means: list[float] = []
    shadow_template_slot_norms: list[float] = []
    shadow_template_usage_entropies: list[float] = []
    shadow_adapter_norms: list[float] = []
    shadow_logit_bias_norms: list[float] = []
    shadow_ln_mod_norms: list[float] = []
    shadow_ancestry_means: list[Tensor] = []
    shadow_regime_means: list[Tensor] = []
    total_bytes = 0.0
    total_nats = 0.0
    total_tokens = 0
    for batch in iter_validation_batches(val_tokens, config.train.val_batch_tokens, config.model.block_size, config.train.max_val_batches):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, amp_dtype):
            outputs = model(batch["input_ids"], targets=batch["targets"], mode="shadow" if shadow_mode else "backbone")
        loss = outputs["loss"] if outputs["loss"] is not None else outputs["gpt_loss"]
        losses.append(float(loss.item()))
        gpt_loss = outputs["gpt_loss"] if outputs["gpt_loss"] is not None else loss
        gpt_losses.append(float(gpt_loss.item()))
        if shadow_mode:
            shadow_gate_means.append(float(outputs["shadow_gate"].mean().item()))
            shadow_gate_raw_means.append(float(outputs["shadow_gate_raw_mean"].mean().item()))
            shadow_active_fracs.append(float(outputs["shadow_active_frac"].mean().item()))
            shadow_bias_norms.append(float(outputs["shadow_bias_norm"].mean().item()))
            shadow_boundary_rates.append(float(outputs["shadow_boundary_rate"].mean().item()))
            shadow_hard_boundary_rates.append(float(outputs["shadow_hard_boundary_rate"].mean().item()))
            shadow_suppression_means.append(float(outputs["shadow_suppression_mean"].mean().item()))
            shadow_token_scale_means.append(float(outputs["shadow_token_scale_mean"].mean().item()))
            shadow_dropout_rates.append(float(outputs["shadow_dropout_rate"].mean().item()))
            shadow_corruption_rates.append(float(outputs["shadow_corruption_rate"].mean().item()))
            shadow_fingerprint_diversities.append(float(outputs["shadow_fingerprint_diversity"].mean().item()))
            shadow_recurrence_tag_means.append(float(outputs["shadow_recurrence_tag_mean"].mean().item()))
            shadow_template_slot_norms.append(float(outputs["shadow_template_slot_norm"].mean().item()))
            shadow_template_usage_entropies.append(float(outputs["shadow_template_usage_entropy"].mean().item()))
            shadow_adapter_norms.append(float(outputs["shadow_adapter_norm"].mean().item()))
            shadow_logit_bias_norms.append(float(outputs["shadow_logit_bias_norm"].mean().item()))
            shadow_ln_mod_norms.append(float(outputs["shadow_ln_mod_norm"].mean().item()))
            shadow_ancestry_means.append(F.softmax(outputs["shadow_ancestry_logits"].float(), dim=-1).mean(dim=(0, 1)).detach().cpu())
            shadow_regime_means.append(outputs["shadow_regime_probs"].mean(dim=(0, 1)).detach().cpu())
        if tokenizer_luts is not None:
            total_bytes += bytes_per_targets(batch["input_ids"], batch["targets"], tokenizer_luts)
            total_nats += float(loss.item()) * batch["targets"].numel()
            total_tokens += int(batch["targets"].numel())
    metrics: dict[str, float] = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_ppl": float(math.exp(min(np.mean(losses), 20.0))) if losses else float("nan"),
        "gpt_only_val_loss": float(np.mean(gpt_losses)) if gpt_losses else float("nan"),
        "gpt_only_val_ppl": float(math.exp(min(np.mean(gpt_losses), 20.0))) if gpt_losses else float("nan"),
    }
    if losses and gpt_losses:
        metrics["val_delta_loss"] = float(np.mean(losses) - np.mean(gpt_losses))
    if shadow_gate_means:
        metrics["shadow_gate_mean"] = float(np.mean(shadow_gate_means))
        metrics["shadow_gate_raw_mean"] = float(np.mean(shadow_gate_raw_means))
        metrics["shadow_active_frac"] = float(np.mean(shadow_active_fracs))
        metrics["shadow_bias_norm"] = float(np.mean(shadow_bias_norms))
        metrics["shadow_boundary_rate"] = float(np.mean(shadow_boundary_rates))
        metrics["shadow_hard_boundary_rate"] = float(np.mean(shadow_hard_boundary_rates))
        metrics["shadow_suppression_mean"] = float(np.mean(shadow_suppression_means))
        metrics["shadow_token_scale_mean"] = float(np.mean(shadow_token_scale_means))
        metrics["shadow_dropout_rate"] = float(np.mean(shadow_dropout_rates))
        metrics["shadow_corruption_rate"] = float(np.mean(shadow_corruption_rates))
        metrics["shadow_fingerprint_diversity"] = float(np.mean(shadow_fingerprint_diversities))
        metrics["shadow_recurrence_tag_mean"] = float(np.mean(shadow_recurrence_tag_means))
        metrics["shadow_template_slot_norm"] = float(np.mean(shadow_template_slot_norms))
        metrics["shadow_template_usage_entropy"] = float(np.mean(shadow_template_usage_entropies))
        metrics["shadow_adapter_norm"] = float(np.mean(shadow_adapter_norms))
        metrics["shadow_logit_bias_norm"] = float(np.mean(shadow_logit_bias_norms))
        metrics["shadow_ln_mod_norm"] = float(np.mean(shadow_ln_mod_norms))
    if shadow_regime_means:
        regime_avg = torch.stack(shadow_regime_means).mean(dim=0)
        for idx, name in enumerate(WebArchaeologyShadow.regime_names):
            metrics[f"shadow_regime_{name}"] = float(regime_avg[idx].item())
    if shadow_ancestry_means:
        ancestry_avg = torch.stack(shadow_ancestry_means).mean(dim=0)
        for idx, name in enumerate(("content", "metadata", "layout", "template")):
            metrics[f"shadow_ancestry_{name}"] = float(ancestry_avg[idx].item())
    if tokenizer_luts is not None and total_bytes > 0 and total_tokens > 0:
        bits_per_token = (total_nats / total_tokens) / math.log(2.0)
        metrics["val_bpb"] = float(bits_per_token * (total_tokens / total_bytes))
    return metrics


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    ordered = " ".join(f"{k}:{v:.4f}" for k, v in sorted(metrics.items()))
    rank0_print(f"{prefix} {ordered}")


def save_checkpoint(path: str | Path, model: HybridGriffinLanguageModel, optimizer: torch.optim.Optimizer | OptimizerBundle | None, config: ExperimentConfig, step: int, extra: dict[str, Any] | None = None) -> None:
    model = unwrap_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "extra": extra or {},
    }
    torch.save(checkpoint, path)
    rank0_print(f"checkpoint_saved:{path}")


def checkpoint_path_for_step(path: str | Path, step: int) -> Path:
    path = Path(path)
    path_str = str(path)
    if "{step}" in path_str:
        return Path(path_str.replace("{step}", str(step)))
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    step_tag = f"_step{step:07d}"
    filename = f"{stem}{step_tag}{suffix}" if suffix else f"{stem}{step_tag}"
    return path.with_name(filename)


def load_checkpoint(path: str | Path, model: HybridGriffinLanguageModel, optimizer: torch.optim.Optimizer | OptimizerBundle | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    model = unwrap_model(model)
    checkpoint = torch.load(path, map_location=map_location)
    checkpoint_state = checkpoint["model_state"]
    current_state = model.state_dict()
    compatible_state: dict[str, Tensor] = {}
    skipped_keys: list[str] = []
    for name, tensor in checkpoint_state.items():
        if name in current_state and current_state[name].shape == tensor.shape:
            compatible_state[name] = tensor
        else:
            skipped_keys.append(name)
    model.load_state_dict(compatible_state, strict=False)
    if skipped_keys:
        preview = ",".join(skipped_keys[:8])
        suffix = "..." if len(skipped_keys) > 8 else ""
        rank0_print(f"checkpoint_model_state:partial loaded:{len(compatible_state)} skipped:{len(skipped_keys)} keys:{preview}{suffix}")
    else:
        rank0_print("checkpoint_model_state:loaded_all")
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            rank0_print("checkpoint_optimizer_state:loaded")
        except ValueError as exc:
            rank0_print(f"checkpoint_optimizer_state:skipped reason:{exc}")
    rank0_print(f"checkpoint_loaded:{path}")
    return checkpoint


def run_training_loop(model: HybridGriffinLanguageModel, train_loader: StreamingTokenLoader, val_tokens: Tensor, config: ExperimentConfig, device: torch.device, amp_dtype: torch.dtype, tokenizer_luts: tuple[Tensor, Tensor, Tensor] | None) -> None:
    configure_model_for_mode(model, config.train.mode)
    base_model = unwrap_model(model)
    master = is_master_process()
    distributed = distributed_is_initialized()
    optimizer = build_optimizer(model, config.train)
    start_step = 0
    if config.train.load_checkpoint:
        ckpt = load_checkpoint(config.train.load_checkpoint, model, optimizer=optimizer, map_location="cpu")
        start_step = int(ckpt.get("step", 0))
    base_model.to(device)
    move_optimizer_state_to_device(optimizer, device)
    base_model.backbone.set_qat(False, config.model.qat_bits, matrix_only=False)
    if master:
        rank0_print(f"mode:{config.train.mode} trainable_params:{count_trainable_parameters(base_model)}")
    best_metric = float("inf")
    best_metric_name = "val_bpb" if tokenizer_luts is not None else "val_loss"
    early_stop_bad_evals = 0
    final_step = start_step
    last_saved_step: int | None = None
    qat_logged = False
    matrix_qat_logged = False
    base_lr = config.train.shadow_lr if config.train.mode == "shadow" else config.train.backbone_lr
    recent_step_times: list[float] = []

    for step in range(start_step, config.train.max_steps):
        step_t0 = time.perf_counter()
        model.train()
        full_qat_active = config.train.mode == "backbone" and config.model.qat_enabled and (step + 1) >= config.train.qat_start_step
        matrix_qat_active = (
            config.train.mode == "backbone"
            and config.model.qat_enabled
            and not full_qat_active
            and config.train.matrix_qat_start_step > 0
            and (step + 1) >= config.train.matrix_qat_start_step
        )
        base_model.backbone.set_qat(full_qat_active or matrix_qat_active, config.model.qat_bits, matrix_only=matrix_qat_active)
        if matrix_qat_active and not matrix_qat_logged and master:
            rank0_print(
                f"matrix_qat_enabled:true qat_bits:{config.model.qat_bits} "
                f"matrix_qat_start_step:{config.train.matrix_qat_start_step} activated_step:{step + 1}"
            )
            matrix_qat_logged = True
        if full_qat_active and not qat_logged and master:
            rank0_print(f"qat_enabled:true qat_bits:{config.model.qat_bits} qat_start_step:{config.train.qat_start_step} activated_step:{step + 1}")
            qat_logged = True
        batch = move_batch_to_device(train_loader.next_batch(config.train.train_batch_tokens, config.model.block_size), device)
        apply_lr_schedule(optimizer, base_lr, lr_scale(step, config.train.warmup_steps))
        maybe_update_muon_momentum(optimizer, config.train, step)
        optimizer.zero_grad(set_to_none=True)
        final_step = step + 1
        with autocast_context(device, amp_dtype):
            outputs = model(batch["input_ids"], targets=batch["targets"], mode=config.train.mode)
            if config.train.mode == "shadow":
                loss, extra = compute_shadow_losses(outputs, config.shadow)
            else:
                loss = outputs["loss"] if outputs["loss"] is not None else outputs["gpt_loss"]
                extra = {}
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss detected at step {step} in mode={config.train.mode}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.train.grad_clip_norm)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - step_t0
        recent_step_times.append(step_time)
        if len(recent_step_times) > max(config.train.log_interval, 8):
            recent_step_times.pop(0)

        if master and (step % config.train.log_interval == 0 or step == config.train.max_steps - 1):
            avg_step_time = float(np.mean(recent_step_times)) if recent_step_times else step_time
            extra["step_ms"] = 1000.0 * avg_step_time
            extra["tok_s"] = float(batch["targets"].numel()) * world_size() / max(avg_step_time, 1e-9)
            if device.type == "cuda":
                extra["cuda_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            if tokenizer_luts is not None:
                train_byte_count = bytes_per_targets(batch["input_ids"], batch["targets"], tokenizer_luts)
                train_bpb = bpb_from_loss(float((outputs["loss"] if outputs["loss"] is not None else outputs["gpt_loss"]).item()), int(batch["targets"].numel()), train_byte_count)
                if train_bpb is not None:
                    extra["train_bpb"] = train_bpb
            log_train_step(step, float(loss.item()), extra)
            if config.train.metrics_csv_path:
                train_metrics = {"loss": float(loss.item())}
                train_metrics.update(extra)
                append_metrics_row(
                    config.train.metrics_csv_path,
                    build_metrics_row("train", step, config.train.mode, train_metrics),
                )

        stop_training = False
        if step % config.train.eval_interval == 0 or step == config.train.max_steps - 1:
            if distributed:
                dist.barrier()
            if master:
                metrics = evaluate(base_model, val_tokens, config, device, amp_dtype, tokenizer_luts)
                print_metrics("eval", metrics)
                if config.train.metrics_csv_path:
                    append_metrics_row(
                        config.train.metrics_csv_path,
                        build_metrics_row("eval", step, config.train.mode, metrics),
                    )
                if config.train.export_probe_every > 0 and config.train.legacy_export_path and (step + 1) % config.train.export_probe_every == 0:
                    probe_baseline_loss, probe_chosen, _ = evaluate_export_candidates(base_model, val_tokens, config, device, amp_dtype)
                    rank0_print(
                        f"export_probe step:{step + 1} plan:{probe_chosen['candidate']['name']} "
                        f"bytes:{probe_chosen['size_bytes']} baseline_loss:{probe_baseline_loss:.4f} "
                        f"export_loss:{probe_chosen['loss']:.4f} delta:{probe_chosen['loss_increase']:.4f}"
                    )
                if config.train.early_stopping_patience > 0:
                    metric_name = "val_bpb" if "val_bpb" in metrics else "val_loss"
                    current_metric = float(metrics[metric_name])
                    improvement = best_metric - current_metric
                    if improvement > config.train.early_stopping_min_delta:
                        best_metric = current_metric
                        best_metric_name = metric_name
                        early_stop_bad_evals = 0
                    else:
                        early_stop_bad_evals += 1
                        rank0_print(f"early_stopping_wait:{early_stop_bad_evals}/{config.train.early_stopping_patience} mode:{config.train.mode} metric:{metric_name} best:{best_metric:.4f} current:{current_metric:.4f}")
                        if early_stop_bad_evals >= config.train.early_stopping_patience:
                            rank0_print(f"early_stopping_triggered step:{step} mode:{config.train.mode} metric:{best_metric_name} best:{best_metric:.4f}")
                            stop_training = True
            if distributed:
                stop_tensor = torch.tensor(int(stop_training), device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
                dist.barrier()
            if stop_training:
                break

        if config.train.checkpoint_path and config.train.save_every > 0 and (step + 1) % config.train.save_every == 0:
            if master:
                save_checkpoint(checkpoint_path_for_step(config.train.checkpoint_path, step + 1), base_model, optimizer, config, step + 1)
                last_saved_step = step + 1
            if distributed:
                dist.barrier()

    if config.train.checkpoint_path and last_saved_step != final_step and master:
        save_checkpoint(checkpoint_path_for_step(config.train.checkpoint_path, final_step), base_model, optimizer, config, final_step)
    if distributed:
        dist.barrier()
    if config.train.legacy_export_path and master:
        export_quantized_checkpoint(base_model, config.train.legacy_export_path, val_tokens, config, device, amp_dtype)
    if distributed:
        dist.barrier()

def checkpoint_path_for_stage(path: str, stage: str) -> str:
    if not path:
        return path
    if "{step}" in path:
        return path.replace("{step}", f"{stage}_{{step}}")
    p = Path(path)
    suffix = "".join(p.suffixes)
    stem = p.name[: -len(suffix)] if suffix else p.name
    filename = f"{stem}_{stage}{suffix}" if suffix else f"{stem}_{stage}"
    return str(p.with_name(filename))


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Single-file Griffin + shadow trainer")
    parser.add_argument("--mode", choices=["backbone", "shadow", "all"], default="backbone")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--train-pattern", default="")
    parser.add_argument("--val-pattern", default="")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--backbone-type", default="griffin_recurrent_gemma")
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qat-bits", type=int, default=6)
    parser.add_argument("--qat-embed-bits", type=int, default=6)
    parser.add_argument("--qat-sensitive-bits", type=int, default=6)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--local-attn-backend", choices=["auto", "sdpa", "flash"], default="auto")
    parser.add_argument("--mlp-mult", type=float, default=4.0)
    parser.add_argument("--recurrent-variant", choices=["rglru", "rglru_relaxed", "legacy_like"], default="rglru")
    parser.add_argument("--recurrent-decay-bias", type=float, default=1.0)
    parser.add_argument("--recurrent-slow-decay-bias", type=float, default=2.5)
    parser.add_argument("--recurrent-chunk-size", type=int, default=128)
    parser.add_argument("--mix-strategy", choices=["learned", "sum", "recurrent_only", "attention_only", "recurrent_delta"], default="learned")
    parser.add_argument("--mix-temperature", type=float, default=1.0)
    parser.add_argument("--recurrent-branch-init", type=float, default=1.0)
    parser.add_argument("--attention-branch-init", type=float, default=0.5)
    parser.add_argument("--train-batch-tokens", type=int, default=131072)
    parser.add_argument("--val-batch-tokens", type=int, default=131072)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--export-calibration-batches", type=int, default=4)
    parser.add_argument("--export-max-loss-increase", type=float, default=0.01)
    parser.add_argument("--export-probe-every", type=int, default=0)
    parser.add_argument("--matrix-qat-start-step", type=int, default=0)
    parser.add_argument("--qat-start-step", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--shadow-lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--backbone-optimizer", choices=["adamw", "muon_adamw"], default="adamw")
    parser.add_argument("--shadow-optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--backbone-embed-lr-scale", type=float, default=0.5)
    parser.add_argument("--backbone-scalar-lr-scale", type=float, default=0.25)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-backend-steps", type=int, default=5)
    parser.add_argument("--muon-momentum-warmup-start", type=float, default=0.85)
    parser.add_argument("--muon-momentum-warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--shadow-window", type=int, default=24)
    parser.add_argument("--shadow-hidden-dim", type=int, default=128)
    parser.add_argument("--shadow-dropout", type=float, default=0.1)
    parser.add_argument("--shadow-control-scale", type=float, default=0.14)
    parser.add_argument("--shadow-gate-bias", type=float, default=-1.5)
    parser.add_argument("--shadow-gate-l1-coef", type=float, default=0.01)
    parser.add_argument("--shadow-gate-threshold", type=float, default=0.55)
    parser.add_argument("--shadow-gate-sharpness", type=float, default=12.0)
    parser.add_argument("--shadow-active-target", type=float, default=0.18)
    parser.add_argument("--shadow-active-excess-coef", type=float, default=0.03)
    parser.add_argument("--shadow-regime-temp", type=float, default=0.7)
    parser.add_argument("--shadow-boundary-scale", type=float, default=0.08)
    parser.add_argument("--shadow-boundary-soft-threshold", type=float, default=0.72)
    parser.add_argument("--shadow-boundary-hard-threshold", type=float, default=0.90)
    parser.add_argument("--shadow-suppression-scale", type=float, default=0.18)
    parser.add_argument("--shadow-suppression-floor", type=float, default=0.78)
    parser.add_argument("--shadow-suppression-l1-coef", type=float, default=0.005)
    parser.add_argument("--shadow-token-dropout-max", type=float, default=0.08)
    parser.add_argument("--shadow-corruption-prob", type=float, default=0.05)
    parser.add_argument("--shadow-fingerprint-buckets", type=int, default=64)
    parser.add_argument("--shadow-fingerprint-scale", type=float, default=0.05)
    parser.add_argument("--shadow-recurrence-scale", type=float, default=0.05)
    parser.add_argument("--shadow-pos-warp-scale", type=float, default=0.05)
    parser.add_argument("--shadow-adapter-scale", type=float, default=0.08)
    parser.add_argument("--shadow-adapter-layers", type=int, default=2)
    parser.add_argument("--shadow-reset-scale", type=float, default=0.06)
    parser.add_argument("--shadow-logit-scale", type=float, default=0.04)
    parser.add_argument("--shadow-consistency-coef", type=float, default=0.05)
    parser.add_argument("--shadow-template-slots", type=int, default=4)
    parser.add_argument("--shadow-template-scale", type=float, default=0.08)
    parser.add_argument("--shadow-ancestry-scale", type=float, default=0.05)
    parser.add_argument("--shadow-ancestry-coef", type=float, default=0.05)
    parser.add_argument("--shadow-ln-mod-scale", type=float, default=0.08)
    parser.add_argument("--shadow-template-norm-coef", type=float, default=1e-4)
    parser.add_argument("--shadow-adapter-norm-coef", type=float, default=5e-5)
    parser.add_argument("--shadow-logit-norm-coef", type=float, default=5e-5)
    parser.add_argument("--shadow-ln-norm-coef", type=float, default=5e-5)
    parser.add_argument("--shadow-ancestry-entropy-coef", type=float, default=0.02)
    parser.add_argument("--shadow-slot-balance-coef", type=float, default=0.02)
    parser.add_argument("--generate-tokens", type=int, default=0)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--legacy-export-path", default="")
    parser.add_argument("--metrics-csv-path", default="")
    add_bool_arg(parser, "bias", False, "use bias terms in linear layers")
    add_bool_arg(parser, "tie-embeddings", True, "tie token embeddings and LM head")
    add_bool_arg(parser, "use-rope", True, "use RoPE in the local attention path")
    add_bool_arg(parser, "use-flash", True, "use PyTorch SDPA fast path on CUDA when available")
    add_bool_arg(parser, "qat", False, "enable fake-quant/QAT during backbone training")
    add_bool_arg(parser, "use-selective-qat", True, "keep embeddings and sensitive recurrent gates at safer QAT bitwidths")
    add_bool_arg(parser, "recurrent-multiscale", True, "use dual-timescale recurrence in the legacy Griffin recurrent path")
    add_bool_arg(parser, "use-recurrent-cache", True, "keep recurrent state between cached decoding calls")
    add_bool_arg(parser, "use-attention-cache", True, "keep local attention KV cache between cached decoding calls")
    add_bool_arg(parser, "use-shadow-recurrent-injection", False, "inject shadow control bias into the recurrent path as well as the token path")
    add_bool_arg(parser, "use-shadow-stream", True, "enable the web archaeology structural shadow stream")
    add_bool_arg(parser, "eval-only", False, "run evaluation only")
    add_bool_arg(parser, "report-bpb", False, "report tokenizer-agnostic bits-per-byte when tokenizer is available")
    add_bool_arg(parser, "skip-sanity-checks", False, "skip local shape/freeze sanity checks")
    add_bool_arg(parser, "compile-model", False, "compile the Griffin model with torch.compile before training")
    args = parser.parse_args()

    model = GriffinModelConfig(
        backbone_type=args.backbone_type,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        tie_embeddings=args.tie_embeddings,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        use_flash=args.use_flash,
        local_attn_backend=args.local_attn_backend,
        qat_enabled=args.qat,
        qat_bits=args.qat_bits,
        qat_embed_bits=args.qat_embed_bits,
        qat_sensitive_bits=args.qat_sensitive_bits,
        use_selective_qat=args.use_selective_qat,
        local_window=args.local_window,
        mlp_mult=args.mlp_mult,
        recurrent_variant=args.recurrent_variant,
        recurrent_decay_bias=args.recurrent_decay_bias,
        recurrent_slow_decay_bias=args.recurrent_slow_decay_bias,
        recurrent_multiscale=args.recurrent_multiscale,
        recurrent_chunk_size=args.recurrent_chunk_size,
        use_recurrent_cache=args.use_recurrent_cache,
        use_attention_cache=args.use_attention_cache,
        mix_strategy=args.mix_strategy,
        mix_temperature=args.mix_temperature,
        recurrent_branch_init=args.recurrent_branch_init,
        attention_branch_init=args.attention_branch_init,
        use_shadow_recurrent_injection=args.use_shadow_recurrent_injection,
    )
    shadow = ShadowConfig(
        enabled=args.use_shadow_stream,
        window=args.shadow_window,
        hidden_dim=args.shadow_hidden_dim,
        dropout=args.shadow_dropout,
        control_scale=args.shadow_control_scale,
        gate_bias=args.shadow_gate_bias,
        gate_l1_coef=args.shadow_gate_l1_coef,
        gate_threshold=args.shadow_gate_threshold,
        gate_sharpness=args.shadow_gate_sharpness,
        active_target=args.shadow_active_target,
        active_excess_coef=args.shadow_active_excess_coef,
        regime_temp=args.shadow_regime_temp,
        boundary_scale=args.shadow_boundary_scale,
        boundary_soft_threshold=args.shadow_boundary_soft_threshold,
        boundary_hard_threshold=args.shadow_boundary_hard_threshold,
        suppression_scale=args.shadow_suppression_scale,
        suppression_floor=args.shadow_suppression_floor,
        suppression_l1_coef=args.shadow_suppression_l1_coef,
        token_dropout_max=args.shadow_token_dropout_max,
        corruption_prob=args.shadow_corruption_prob,
        fingerprint_buckets=args.shadow_fingerprint_buckets,
        fingerprint_scale=args.shadow_fingerprint_scale,
        recurrence_scale=args.shadow_recurrence_scale,
        pos_warp_scale=args.shadow_pos_warp_scale,
        adapter_scale=args.shadow_adapter_scale,
        adapter_layers=args.shadow_adapter_layers,
        reset_scale=args.shadow_reset_scale,
        logit_scale=args.shadow_logit_scale,
        consistency_coef=args.shadow_consistency_coef,
        template_slots=args.shadow_template_slots,
        template_scale=args.shadow_template_scale,
        ancestry_scale=args.shadow_ancestry_scale,
        ancestry_coef=args.shadow_ancestry_coef,
        ln_mod_scale=args.shadow_ln_mod_scale,
        template_norm_coef=args.shadow_template_norm_coef,
        adapter_norm_coef=args.shadow_adapter_norm_coef,
        logit_norm_coef=args.shadow_logit_norm_coef,
        ln_norm_coef=args.shadow_ln_norm_coef,
        ancestry_entropy_coef=args.shadow_ancestry_entropy_coef,
        slot_balance_coef=args.shadow_slot_balance_coef,
    )
    train = TrainConfig(
        mode=args.mode,
        device_preference=args.device,
        seed=args.seed,
        run_id=args.run_id,
        data_path=args.data_path,
        train_pattern=args.train_pattern,
        val_pattern=args.val_pattern,
        tokenizer_path=args.tokenizer_path,
        train_batch_tokens=args.train_batch_tokens,
        val_batch_tokens=args.val_batch_tokens,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        save_every=args.save_every,
        max_val_batches=args.max_val_batches,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        backbone_lr=args.backbone_lr,
        shadow_lr=args.shadow_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        backbone_optimizer=args.backbone_optimizer,
        shadow_optimizer=args.shadow_optimizer,
        backbone_embed_lr_scale=args.backbone_embed_lr_scale,
        backbone_scalar_lr_scale=args.backbone_scalar_lr_scale,
        muon_momentum=args.muon_momentum,
        muon_backend_steps=args.muon_backend_steps,
        muon_momentum_warmup_start=args.muon_momentum_warmup_start,
        muon_momentum_warmup_steps=args.muon_momentum_warmup_steps,
        checkpoint_path=args.checkpoint_path,
        load_checkpoint=args.load_checkpoint,
        eval_only=args.eval_only,
        report_bpb=args.report_bpb,
        legacy_export_path=args.legacy_export_path,
        metrics_csv_path=args.metrics_csv_path,
        export_calibration_batches=args.export_calibration_batches,
        export_max_loss_increase=args.export_max_loss_increase,
        export_probe_every=args.export_probe_every,
        matrix_qat_start_step=args.matrix_qat_start_step,
        qat_start_step=args.qat_start_step,
        generate_tokens=args.generate_tokens,
        compile_model=args.compile_model,
    )
    config = ExperimentConfig(model=model, shadow=shadow, train=train)
    setattr(config.train, "skip_sanity_checks", args.skip_sanity_checks)
    setattr(config.train, "benchmark_steps", args.benchmark_steps)
    return config


def log_trainable_summary(model: HybridGriffinLanguageModel, mode: str) -> None:
    configure_model_for_mode(model, mode)
    backbone_trainable = count_trainable_parameters(model.backbone)
    shadow_trainable = count_trainable_parameters(model.shadow_stream)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    preview = ",".join(trainable_names[:8])
    suffix = "..." if len(trainable_names) > 8 else ""
    print(
        f"trainability mode:{mode} "
        f"backbone_trainable:{backbone_trainable} "
        f"shadow_trainable:{shadow_trainable} "
        f"trainable_tensors:{len(trainable_names)} "
        f"names:{preview}{suffix}"
    )


def run_sanity_checks(model: HybridGriffinLanguageModel, config: ExperimentConfig, device: torch.device) -> None:
    seq_len = min(8, int(config.model.block_size))
    batch_size = 2
    with torch.no_grad():
        model.eval()
        sample_ids = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        sample_targets = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        backbone_out = model(sample_ids, targets=sample_targets, mode="backbone")
        assert backbone_out["logits"] is not None
        assert backbone_out["hidden_states"] is not None
        assert backbone_out["loss"] is not None
        assert backbone_out["logits"].shape == (batch_size, seq_len, int(config.model.vocab_size))
        assert backbone_out["hidden_states"].shape == (batch_size, seq_len, int(config.model.n_embd))
        assert torch.isfinite(backbone_out["logits"]).all()
        assert torch.isfinite(backbone_out["loss"])
        cached_prefix = model.backbone(sample_ids[:, :-1], return_hidden=True, use_cache=True)
        cache = cached_prefix["cache"]
        assert cache is not None
        assert cache.position == seq_len - 1
        assert len(cache.layers) == int(config.model.n_layer)
        for layer_cache in cache.layers:
            if config.model.use_recurrent_cache:
                assert layer_cache.recurrent_state is not None
                assert layer_cache.recurrent_state.shape == (batch_size, int(config.model.n_embd))
            if config.model.use_attention_cache:
                assert layer_cache.attn_k is not None and layer_cache.attn_v is not None
                assert layer_cache.attn_k.shape[0] == batch_size
                assert layer_cache.attn_k.shape[2] == int(config.model.n_head)
                assert layer_cache.attn_k.shape[-1] == int(config.model.n_embd) // int(config.model.n_head)
        cached_last = model.backbone(sample_ids[:, -1:], return_hidden=True, cache=cache, use_cache=True)
        assert cached_last["cache"] is not None and cached_last["cache"].position == seq_len
        assert torch.allclose(
            backbone_out["logits"][:, -1, :].float(),
            cached_last["logits"][:, -1, :].float(),
            atol=1e-3,
            rtol=1e-3,
        ), "cached decoding diverged from full-sequence forward"
        if config.shadow.enabled:
            configure_model_for_mode(model, "shadow")
            assert count_trainable_parameters(model.backbone) == 0
            assert count_trainable_parameters(model.shadow_stream) > 0
            shadow_out = model(sample_ids, targets=sample_targets, mode="shadow")
            assert shadow_out["logits"] is not None
            assert shadow_out["hidden_states"] is not None
            assert shadow_out["loss"] is not None
            assert shadow_out["logits"].shape == (batch_size, seq_len, int(config.model.vocab_size))
            assert shadow_out["hidden_states"].shape == (batch_size, seq_len, int(config.model.n_embd))
            assert torch.isfinite(shadow_out["logits"]).all()
            assert torch.isfinite(shadow_out["loss"])
        configure_model_for_mode(model, "backbone")
        assert count_trainable_parameters(model.backbone) > 0
        print(
            f"sanity_checks:ok seq_len:{seq_len} "
            f"logits_shape:{tuple(backbone_out['logits'].shape)} "
            f"hidden_shape:{tuple(backbone_out['hidden_states'].shape)} "
            f"cache_layers:{len(cache.layers)}"
        )


@torch.no_grad()
def benchmark_backbone(model: HybridGriffinLanguageModel, config: ExperimentConfig, device: torch.device, amp_dtype: torch.dtype, steps: int) -> None:
    if steps <= 0:
        return
    model = unwrap_model(model)
    model.eval()
    batch_tokens = min(config.train.train_batch_tokens, config.model.block_size * 8)
    batch_seqs = max(1, batch_tokens // config.model.block_size)
    input_ids = torch.randint(0, int(config.model.vocab_size), (batch_seqs, int(config.model.block_size)), device=device)
    warmup = min(3, steps)
    for _ in range(warmup):
        with autocast_context(device, amp_dtype):
            model.backbone(input_ids, return_hidden=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        with autocast_context(device, amp_dtype):
            model.backbone(input_ids, return_hidden=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    tok_s = steps * input_ids.numel() / max(elapsed, 1e-9)
    step_ms = 1000.0 * elapsed / max(steps, 1)
    pieces = [
        f"benchmark_steps:{steps}",
        f"backend:{config.model.local_attn_backend}",
        f"recurrent_variant:{config.model.recurrent_variant}",
        f"chunk_size:{config.model.recurrent_chunk_size}",
        f"step_ms:{step_ms:.2f}",
        f"tok_s:{tok_s:.1f}",
    ]
    if device.type == "cuda":
        pieces.append(f"cuda_mem_gb:{torch.cuda.max_memory_allocated(device) / (1024**3):.3f}")
    rank0_print("benchmark " + " ".join(pieces))


def main() -> None:
    config = parse_args()
    with setup_run_logging(config.train.run_id):
        code = Path(__file__).read_text(encoding="utf-8")
        print(code)
        print("=" * 100)
        print(f"Running Python {sys.version}")
        print(f"Running PyTorch {torch.__version__}")
        gpu_report = ""
        try:
            if torch.cuda.is_available():
                gpu_report = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout
        except Exception as exc:
            gpu_report = f"nvidia_smi_unavailable:{exc}"
        if gpu_report:
            print(gpu_report)
        print("=" * 100)

        seed_everything(config.train.seed)
        device = get_device(config.train.device_preference)
        amp_dtype = get_default_dtype_for_device(device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        tokenizer_luts = build_sentencepiece_luts(config.train.tokenizer_path, config.model.vocab_size) if config.train.report_bpb else None
        shadow_metadata = build_shadow_vocab_metadata(config.train.tokenizer_path, config.model.vocab_size)
        log_device_info(device, amp_dtype, supports_fast_attention(device, config.model.use_flash))
        print(
            f"griffin_config backbone_type:{config.model.backbone_type} "
            f"local_attn_backend:{config.model.local_attn_backend} "
            f"flash_attn_available:{flash_attn_available()} "
            f"recurrent_variant:{config.model.recurrent_variant} "
            f"recurrent_chunk_size:{config.model.recurrent_chunk_size} "
            f"use_recurrent_cache:{config.model.use_recurrent_cache} "
            f"use_attention_cache:{config.model.use_attention_cache} "
            f"mix_strategy:{config.model.mix_strategy}"
        )
        print(json.dumps(asdict(config), indent=2))

        train_loader = StreamingTokenLoader(config.train.resolved_train_pattern())
        val_tokens = load_validation_tokens(config.train.resolved_val_pattern(), config.model.block_size)
        model = HybridGriffinLanguageModel(config.model, config.shadow, shadow_metadata)
        print(
            f"model_params backbone:{sum(p.numel() for p in model.backbone.parameters())} "
            f"shadow:{sum(p.numel() for p in model.shadow_stream.parameters())} "
            f"total:{sum(p.numel() for p in model.parameters())}"
        )
        model.to(device)
        if not getattr(config.train, "skip_sanity_checks", False):
            run_sanity_checks(model, config, device)
        benchmark_backbone(model, config, device, amp_dtype, int(getattr(config.train, "benchmark_steps", 0)))

        if config.train.eval_only:
            if config.train.mode == "all":
                raise ValueError("--eval-only is only supported with --mode backbone or --mode shadow, not --mode all.")
            if config.train.load_checkpoint:
                load_checkpoint(config.train.load_checkpoint, model, map_location="cpu")
            log_trainable_summary(model, config.train.mode)
            model.to(device)
            metrics = evaluate(model, val_tokens, config, device, amp_dtype, tokenizer_luts)
            print_metrics("eval_only", metrics)
            return

        if config.train.mode == "all":
            backbone_train = replace(
                config.train,
                mode="backbone",
                load_checkpoint=config.train.load_checkpoint,
                checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "backbone"),
                legacy_export_path="",
            )
            backbone_config = replace(config, train=backbone_train)
            print("pipeline_stage:backbone")
            log_trainable_summary(model, "backbone")
            run_training_loop(model, train_loader, val_tokens, backbone_config, device, amp_dtype, tokenizer_luts)

            shadow_train = replace(
                config.train,
                mode="shadow",
                load_checkpoint="",
                checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "shadow"),
            )
            shadow_config = replace(config, train=shadow_train)
            print("pipeline_stage:shadow")
            log_trainable_summary(model, "shadow")
            run_training_loop(model, train_loader, val_tokens, shadow_config, device, amp_dtype, tokenizer_luts)
        else:
            log_trainable_summary(model, config.train.mode)
            run_training_loop(model, train_loader, val_tokens, config, device, amp_dtype, tokenizer_luts)

        if config.train.generate_tokens > 0:
            model.eval()
            model.to(device)
            prompt = torch.zeros((1, min(8, config.model.block_size)), dtype=torch.long, device=device)
            generated = model.backbone.generate(prompt, max_new_tokens=config.train.generate_tokens)
            print(f"generated_tokens:{generated[0].tolist()}")

# Legacy Griffin is the default backbone path below. The experimental
# RecurrentGemma-style implementation remains above only as shared utility
# context until it can be removed cleanly in a later simplification pass.

class LegacyGriffinLocalAttention(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.window = max(2, min(config.local_window, config.block_size))
        self.use_flash = config.use_flash
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base) if config.use_rope else None
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.use_selective_qat = config.use_selective_qat
        self.qat_matrix_only = False

    def _local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[:, None] - positions[None, :]
        allowed = (rel >= 0) & (rel < self.window)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
        return mask.masked_fill(allowed, 0.0)

    def _manual_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        scale = 1.0 / math.sqrt(q.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        att = att + mask[None, None, :, :]
        att = F.softmax(att.float(), dim=-1).to(dtype=q.dtype)
        att = self.attn_dropout(att)
        return att @ v

    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor | None:
        if _flash_attn_func is None or not self.use_flash:
            return None
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        try:
            y = _flash_attn_func(
                q_bshd,
                k_bshd,
                v_bshd,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window - 1, 0),
            )
        except TypeError:
            try:
                y = _flash_attn_func(
                    q_bshd,
                    k_bshd,
                    v_bshd,
                    self.dropout if self.training else 0.0,
                    causal=True,
                    window_size=(self.window - 1, 0),
                )
            except Exception:
                return None
        except Exception:
            return None
        if y is None or not torch.isfinite(y).all():
            return None
        return y.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        qat_on = self.training and self.qat_enabled
        q = linear_with_optional_qat(x, self.q_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = linear_with_optional_qat(x, self.k_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = linear_with_optional_qat(x, self.v_proj, qat_on, self.qat_bits).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, q.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        mask = self._local_mask(seq_len, x.device)
        y = self._flash_attention(q, k, v)
        if y is not None:
            pass
        elif supports_fast_attention(x.device, self.use_flash):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask[None, None, :, :],
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            y = self._manual_attention(q, k, v, mask)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits))


class LegacyGriffinRecurrentPath(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.decay_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.recurrent_multiscale = bool(config.recurrent_multiscale)
        self.slow_decay_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) if self.recurrent_multiscale else None
        self.slow_gate_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) if self.recurrent_multiscale else None
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.decay_bias = nn.Parameter(torch.full((config.n_embd,), float(config.recurrent_decay_bias)))
        self.slow_decay_bias = nn.Parameter(torch.full((config.n_embd,), float(config.recurrent_slow_decay_bias))) if self.recurrent_multiscale else None
        self.timescale_mix = nn.Parameter(torch.zeros(config.n_embd)) if self.recurrent_multiscale else None
        self.chunk_size = max(0, int(config.recurrent_chunk_size))
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.qat_sensitive_bits = clamp_quant_bits(config.qat_sensitive_bits)
        self.use_selective_qat = config.use_selective_qat
        self.qat_matrix_only = False

    def forward(self, x: Tensor) -> Tensor:
        qat_on = self.training and self.qat_enabled
        inp = F.silu(linear_with_optional_qat(x, self.in_proj, qat_on, self.qat_bits)).float()
        sensitive_qat_on = qat_on and not self.qat_matrix_only
        sensitive_bits = self.qat_sensitive_bits if self.use_selective_qat else self.qat_bits
        decay = torch.sigmoid(linear_with_optional_qat(x, self.decay_proj, sensitive_qat_on, sensitive_bits).float() + self.decay_bias)
        gate = torch.sigmoid(linear_with_optional_qat(x, self.gate_proj, sensitive_qat_on, sensitive_bits).float())
        recurrent_inputs = (1.0 - decay) * inp
        states, _ = diagonal_linear_scan(decay, recurrent_inputs, initial_state=None, chunk_size=self.chunk_size)
        fast_out = gate * states
        y = fast_out
        if self.recurrent_multiscale and self.slow_decay_proj is not None and self.slow_gate_proj is not None and self.slow_decay_bias is not None and self.timescale_mix is not None:
            slow_decay = torch.sigmoid(linear_with_optional_qat(x, self.slow_decay_proj, sensitive_qat_on, sensitive_bits).float() + self.slow_decay_bias)
            slow_gate = torch.sigmoid(linear_with_optional_qat(x, self.slow_gate_proj, sensitive_qat_on, sensitive_bits).float())
            slow_inputs = (1.0 - slow_decay) * inp
            slow_states, _ = diagonal_linear_scan(slow_decay, slow_inputs, initial_state=None, chunk_size=self.chunk_size)
            slow_out = slow_gate * slow_states
            mix = torch.sigmoid(self.timescale_mix).to(dtype=fast_out.dtype)[None, None, :]
            y = mix * fast_out + (1.0 - mix) * slow_out
        y = y.to(dtype=x.dtype)
        y = linear_with_optional_qat(y, self.out_proj, qat_on, self.qat_bits)
        return self.resid_dropout(y)


class LegacyGriffinMLP(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        hidden = max(config.n_embd, int(round(config.n_embd * config.mlp_mult)))
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.qat_sensitive_bits = clamp_quant_bits(config.qat_sensitive_bits)
        self.use_selective_qat = config.use_selective_qat
        self.qat_matrix_only = False

    def forward(self, x: Tensor) -> Tensor:
        qat_on = self.training and self.qat_enabled
        gate_bits = self.qat_sensitive_bits if self.use_selective_qat else self.qat_bits
        gate = F.silu(linear_with_optional_qat(x, self.gate_proj, qat_on and not self.qat_matrix_only, gate_bits))
        value = linear_with_optional_qat(x, self.up_proj, qat_on, self.qat_bits)
        out = linear_with_optional_qat(gate * value, self.down_proj, qat_on, self.qat_bits)
        return self.dropout(out)


class LegacyGriffinBlock(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.mix_strategy = config.mix_strategy
        self.temporal_norm = RMSNorm(config.n_embd)
        self.local_attn = LegacyGriffinLocalAttention(config)
        self.recurrent = LegacyGriffinRecurrentPath(config)
        self.temporal_mix = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.channel_norm = RMSNorm(config.n_embd)
        self.channel_mlp = LegacyGriffinMLP(config)
        self.mix_temperature = max(float(config.mix_temperature), 1e-3)
        self.recurrent_scale = nn.Parameter(torch.full((config.n_embd,), float(config.recurrent_branch_init)))
        self.attention_scale = nn.Parameter(torch.full((config.n_embd,), float(config.attention_branch_init)))
        nn.init.zeros_(self.temporal_mix.weight)
        nn.init.zeros_(self.temporal_mix.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.temporal_norm(x)
        recurrent_out = self.recurrent(h) * self.recurrent_scale.to(dtype=h.dtype)[None, None, :]
        attn_out = self.local_attn(h) * self.attention_scale.to(dtype=h.dtype)[None, None, :]
        if self.mix_strategy == "recurrent_only":
            temporal_out = recurrent_out
        elif self.mix_strategy == "attention_only":
            temporal_out = attn_out
        elif self.mix_strategy == "sum":
            temporal_out = recurrent_out + attn_out
        else:
            mix = torch.sigmoid(self.temporal_mix(h) / self.mix_temperature)
            if self.mix_strategy == "recurrent_delta":
                temporal_out = recurrent_out + mix * attn_out
            else:
                temporal_out = mix * attn_out + (1.0 - mix) * recurrent_out
        x = x + self.temporal_dropout(temporal_out)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class LegacyGriffinBackbone(nn.Module):
    def __init__(self, config: GriffinModelConfig):
        super().__init__()
        self.config = config
        self.qat_enabled = config.qat_enabled
        self.qat_bits = clamp_quant_bits(config.qat_bits)
        self.qat_embed_bits = clamp_quant_bits(config.qat_embed_bits)
        self.qat_sensitive_bits = clamp_quant_bits(config.qat_sensitive_bits)
        self.use_selective_qat = config.use_selective_qat
        self.qat_matrix_only = False
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([LegacyGriffinBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_qat(self, enabled: bool, bits: int | None = None, matrix_only: bool = False) -> None:
        self.qat_enabled = enabled
        self.qat_matrix_only = matrix_only
        if bits is not None:
            self.qat_bits = clamp_quant_bits(bits)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "qat_enabled"):
                module.qat_enabled = enabled
            if hasattr(module, "qat_matrix_only"):
                module.qat_matrix_only = matrix_only
            if bits is not None and hasattr(module, "qat_bits"):
                module.qat_bits = self.qat_bits

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        return_hidden: bool = True,
        control_bias: Tensor | None = None,
        token_scale: Tensor | None = None,
        pos_bias: Tensor | None = None,
        adapter_bias: Tensor | None = None,
        adapter_layers: int = 0,
        ln_gain: Tensor | None = None,
        ln_bias: Tensor | None = None,
        cache=None,
        use_cache: bool = False,
    ) -> dict[str, Tensor | None]:
        del cache, use_cache
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")
        qat_on = self.training and self.qat_enabled
        embed_qat_on = qat_on and not self.qat_matrix_only
        embed_bits = self.qat_embed_bits if self.use_selective_qat else max(self.qat_bits, 6)
        tok = embedding_with_optional_qat(input_ids, self.wte, embed_qat_on, embed_bits)
        if self.wpe is not None:
            pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            pos_emb = embedding_with_optional_qat(pos, self.wpe, embed_qat_on, embed_bits)
            x = tok + pos_emb[None, :, :]
        else:
            x = tok
        if pos_bias is not None:
            x = x + pos_bias
        if control_bias is not None:
            x = x + control_bias
        if token_scale is not None:
            x = x * token_scale.unsqueeze(-1)
        x = self.drop(x)
        apply_adapter_from = max(len(self.blocks) - max(adapter_layers, 0), 0)
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if adapter_bias is not None and block_idx >= apply_adapter_from:
                x = x + adapter_bias
        hidden = self.ln_f(x)
        if ln_gain is not None:
            hidden = hidden * (1.0 + ln_gain)
        if ln_bias is not None:
            hidden = hidden + ln_bias
        logits = linear_with_optional_qat(hidden, self.lm_head, embed_qat_on, embed_bits)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
        return {"logits": logits, "loss": loss, "hidden_states": hidden if return_hidden else None, "cache": None}

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        out = input_ids
        for _ in range(max_new_tokens):
            idx = out[:, -self.config.block_size :]
            logits = self(idx, return_hidden=False)["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out


class LegacyHybridGriffinLanguageModel(nn.Module):
    def __init__(self, model_config: GriffinModelConfig, shadow_config: ShadowConfig, shadow_metadata: dict[str, Tensor] | None = None):
        super().__init__()
        self.backbone = LegacyGriffinBackbone(model_config)
        self.model_config = model_config
        self.shadow_config = shadow_config
        self.shadow_stream = WebArchaeologyShadow(model_config.n_embd, model_config.vocab_size, shadow_config, shadow_metadata)

    def forward_backbone(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        out = self.backbone(input_ids, targets=targets, return_hidden=True)
        return {"logits": out["logits"], "loss": out["loss"], "hidden_states": out["hidden_states"], "gpt_logits": out["logits"], "gpt_loss": out["loss"]}

    def forward_shadow(self, input_ids: Tensor, targets: Tensor | None = None) -> dict[str, Tensor | None]:
        with torch.no_grad():
            base_out = self.backbone(input_ids, targets=None, return_hidden=True)
            gpt_logits = base_out["logits"]
            hidden = base_out["hidden_states"]
        assert hidden is not None
        shadow_out = self.shadow_stream(input_ids, gpt_logits.detach(), hidden.detach())
        controlled_out = self.backbone(
            shadow_out["augmented_input_ids"],
            targets=None,
            return_hidden=True,
            control_bias=shadow_out["control_bias"],
            token_scale=shadow_out["token_scale"],
            pos_bias=shadow_out["pos_bias"],
            adapter_bias=shadow_out["adapter_bias"],
            adapter_layers=self.shadow_config.adapter_layers,
            ln_gain=shadow_out["ln_gain"],
            ln_bias=shadow_out["ln_bias"],
        )
        logits = controlled_out["logits"] + shadow_out["logit_bias"]
        outputs: dict[str, Tensor | None] = {
            "logits": logits,
            "loss": None,
            "gpt_logits": gpt_logits,
            "gpt_loss": None,
            "hidden_states": controlled_out["hidden_states"],
            "shadow_gate": shadow_out["gate"],
            "shadow_gate_raw_mean": shadow_out["gate_raw_mean"],
            "shadow_active_frac": shadow_out["active_frac"],
            "shadow_bias_norm": shadow_out["bias_norm"],
            "shadow_boundary_rate": shadow_out["boundary_rate"],
            "shadow_hard_boundary_rate": shadow_out["hard_boundary_rate"],
            "shadow_suppression_mean": shadow_out["suppression_mean"],
            "shadow_token_scale_mean": shadow_out["token_scale"].mean(dim=1),
            "shadow_dropout_rate": shadow_out["dropout_rate"],
            "shadow_corruption_rate": shadow_out["corruption_rate"],
            "shadow_fingerprint_diversity": shadow_out["fingerprint_diversity"],
            "shadow_recurrence_tag_mean": shadow_out["recurrence_tag_mean"],
            "shadow_template_slot_norm": shadow_out["template_slot_norm"],
            "shadow_template_usage_entropy": shadow_out["template_usage_entropy"],
            "shadow_adapter_norm": shadow_out["adapter_norm"],
            "shadow_logit_bias_norm": shadow_out["logit_bias_norm"],
            "shadow_ln_mod_norm": shadow_out["ln_mod_norm"],
            "shadow_regime_probs": shadow_out["regime_probs"],
            "shadow_consistency_logits": shadow_out["consistency_logits"],
            "shadow_ancestry_logits": shadow_out["ancestry_logits"],
        }
        if targets is not None:
            outputs["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["gpt_loss"] = F.cross_entropy(gpt_logits.reshape(-1, gpt_logits.size(-1)).float(), targets.reshape(-1), reduction="mean")
            outputs["shadow_target_logprobs"] = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["gpt_target_logprobs"] = F.log_softmax(gpt_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            outputs["shadow_consistency_loss"] = F.cross_entropy(
                shadow_out["consistency_logits"][:, :-1, :].reshape(-1, shadow_out["consistency_logits"].size(-1)).float(),
                shadow_out["regime_probs"][:, 1:, :].argmax(dim=-1).reshape(-1),
                reduction="mean",
            )
            outputs["shadow_ancestry_loss"] = F.cross_entropy(
                shadow_out["ancestry_logits"].reshape(-1, shadow_out["ancestry_logits"].size(-1)).float(),
                shadow_out["ancestry_targets"].reshape(-1),
                reduction="mean",
            )
        return outputs

    def forward(self, input_ids: Tensor, targets: Tensor | None = None, mode: str = "backbone") -> dict[str, Tensor | None]:
        if mode == "shadow":
            return self.forward_shadow(input_ids, targets=targets)
        return self.forward_backbone(input_ids, targets=targets)


def run_sanity_checks_old(model: LegacyHybridGriffinLanguageModel, config: ExperimentConfig, device: torch.device) -> None:
    seq_len = min(8, int(config.model.block_size))
    batch_size = 2
    with torch.no_grad():
        model.eval()
        sample_ids = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        sample_targets = torch.randint(0, int(config.model.vocab_size), (batch_size, seq_len), device=device)
        backbone_out = model(sample_ids, targets=sample_targets, mode="backbone")
        assert backbone_out["logits"] is not None and backbone_out["hidden_states"] is not None
        assert backbone_out["logits"].shape == (batch_size, seq_len, int(config.model.vocab_size))
        assert backbone_out["hidden_states"].shape == (batch_size, seq_len, int(config.model.n_embd))
        if config.shadow.enabled:
            configure_model_for_mode(model, "shadow")
            assert count_trainable_parameters(model.backbone) == 0
            assert count_trainable_parameters(model.shadow_stream) > 0
            shadow_out = model(sample_ids, targets=sample_targets, mode="shadow")
            assert shadow_out["logits"] is not None and shadow_out["hidden_states"] is not None
        configure_model_for_mode(model, "backbone")
        rank0_print(
            f"legacy_sanity_checks:ok seq_len:{seq_len} "
            f"logits_shape:{tuple(backbone_out['logits'].shape)} "
            f"hidden_shape:{tuple(backbone_out['hidden_states'].shape)}"
        )


def main() -> None:
    config = parse_args()
    config.model.backbone_type = "griffin_legacy_simple"
    config.model.local_attn_backend = "sdpa"
    config.model.recurrent_variant = "legacy_simple"
    config.model.use_recurrent_cache = False
    config.model.use_attention_cache = False
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    log_context = setup_run_logging(config.train.run_id) if is_master_process() else contextlib.nullcontext(None)
    try:
        with log_context:
            code = Path(__file__).read_text(encoding="utf-8")
            rank0_print(code)
            rank0_print("=" * 100)
            rank0_print(f"Running Python {sys.version}")
            rank0_print(f"Running PyTorch {torch.__version__}")
            gpu_report = ""
            try:
                if torch.cuda.is_available():
                    gpu_report = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout
            except Exception as exc:
                gpu_report = f"nvidia_smi_unavailable:{exc}"
            if gpu_report:
                rank0_print(gpu_report)
            rank0_print("=" * 100)

            seed_everything(config.train.seed + global_rank())
            device = torch.device("cuda", local_rank) if distributed else get_device(config.train.device_preference)
            amp_dtype = get_default_dtype_for_device(device)
            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            tokenizer_luts = build_sentencepiece_luts(config.train.tokenizer_path, config.model.vocab_size) if config.train.report_bpb else None
            shadow_metadata = build_shadow_vocab_metadata(config.train.tokenizer_path, config.model.vocab_size)
            log_device_info(device, amp_dtype, supports_fast_attention(device, config.model.use_flash))
            rank0_print(
                f"griffin_config backbone_type:{config.model.backbone_type} "
                f"local_attn_backend:{config.model.local_attn_backend} "
                f"flash_attn_available:{flash_attn_available()} "
                f"recurrent_variant:{config.model.recurrent_variant} "
                f"use_recurrent_cache:{config.model.use_recurrent_cache} "
                f"use_attention_cache:{config.model.use_attention_cache}"
            )
            rank0_print(json.dumps(asdict(config), indent=2))

            train_loader = StreamingTokenLoader(config.train.resolved_train_pattern(), rank=global_rank(), world_size=world_size())
            val_tokens = load_validation_tokens(config.train.resolved_val_pattern(), config.model.block_size)
            base_model = LegacyHybridGriffinLanguageModel(config.model, config.shadow, shadow_metadata)
            rank0_print(
                f"model_params backbone:{sum(p.numel() for p in base_model.backbone.parameters())} "
                f"shadow:{sum(p.numel() for p in base_model.shadow_stream.parameters())} "
                f"total:{sum(p.numel() for p in base_model.parameters())}"
            )
            base_model.to(device)
            if is_master_process() and not getattr(config.train, "skip_sanity_checks", False):
                run_sanity_checks_old(base_model, config, device)
            if is_master_process():
                benchmark_backbone(base_model, config, device, amp_dtype, int(getattr(config.train, "benchmark_steps", 0)))
            if config.train.compile_model:
                if not hasattr(torch, "compile"):
                    raise RuntimeError("torch.compile is not available in this PyTorch build")
                base_model = torch.compile(base_model, dynamic=False)
                rank0_print("compile_model:true")
            model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

            if config.train.eval_only:
                if config.train.mode == "all":
                    raise ValueError("--eval-only is only supported with --mode backbone or --mode shadow, not --mode all.")
                if config.train.load_checkpoint:
                    load_checkpoint(config.train.load_checkpoint, model, map_location="cpu")
                metrics = evaluate(unwrap_model(model), val_tokens, config, device, amp_dtype, tokenizer_luts)
                print_metrics("eval_only", metrics)
                return

            if config.train.mode == "all":
                backbone_train = replace(
                    config.train,
                    mode="backbone",
                    load_checkpoint=config.train.load_checkpoint,
                    checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "backbone"),
                    legacy_export_path="",
                )
                backbone_config = replace(config, train=backbone_train)
                rank0_print("pipeline_stage:backbone")
                run_training_loop(model, train_loader, val_tokens, backbone_config, device, amp_dtype, tokenizer_luts)

                shadow_train = replace(
                    config.train,
                    mode="shadow",
                    load_checkpoint="",
                    checkpoint_path=checkpoint_path_for_stage(config.train.checkpoint_path, "shadow"),
                )
                shadow_config = replace(config, train=shadow_train)
                rank0_print("pipeline_stage:shadow")
                run_training_loop(model, train_loader, val_tokens, shadow_config, device, amp_dtype, tokenizer_luts)
            else:
                run_training_loop(model, train_loader, val_tokens, config, device, amp_dtype, tokenizer_luts)
    finally:
        if distributed_is_initialized():
            dist.barrier()
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
