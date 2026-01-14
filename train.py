"""
Pre-training script with Flash Attention, DeepSpeed, and Transformer Engine.

Requirements:
    pip install torch deepspeed transformer-engine flash-attn datasets tiktoken wandb

Usage:
    deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset, DataLoader

# Transformer Engine (FP8 mixed precision)
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Flash Attention 2
from flash_attn import flash_attn_func

# Flash Attention 3 (loaded lazily via kernels library)
_fa3_module = None

def _get_fa3():
    global _fa3_module
    if _fa3_module is None:
        from kernels import get_kernel
        _fa3_module = get_kernel("kernels-community/flash-attn3")
    return _fa3_module


# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class ModelConfig:
    vocab_size: int = 50304       # GPT-2 vocab size (padded for efficiency)
    max_seq_len: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None  # None = MHA, else GQA
    d_model: int = 1024
    d_ff: Optional[int] = None        # None = 4 * d_model
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    use_flash_attn: bool = True
    use_flash_attn_3: bool = False    # Flash Attention 3 via kernels (Hopper GPU)
    use_te: bool = True               # Use Transformer Engine layers

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.head_dim = self.d_model // self.n_heads


# =============================================================================
# Rotary Positional Embedding (RoPE)
# =============================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to query/key tensors. x: (B, n_heads, S, head_dim)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out


# =============================================================================
# Grouped-Query Attention with Flash Attention
# =============================================================================
class GQAFlashAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_flash = config.use_flash_attn
        self.use_flash_3 = config.use_flash_attn_3

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=config.bias)

        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.attn_dropout = config.dropout

        # Load Flash Attention 3 kernel
        if self.use_flash_3:
            fa3 = _get_fa3()
            self.fa3_func = fa3.flash_attn_func

    def forward(self, x: torch.Tensor):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # RoPE (applied in (B, heads, S, D) format)
        cos, sin = self.rotary(S)
        q = q.transpose(1, 2)  # (B, n_heads, S, D)
        k = k.transpose(1, 2)  # (B, n_kv_heads, S, D)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.use_flash_3:
            # Flash Attention 3: native GQA via pack_gqa (no KV expansion needed)
            q = q.transpose(1, 2).contiguous()   # (B, S, n_heads, D)
            k = k.transpose(1, 2).contiguous()   # (B, S, n_kv_heads, D)
            v = v.contiguous()                    # (B, S, n_kv_heads, D)
            attn_out = self.fa3_func(q, k, v, causal=True, pack_gqa=True)
        elif self.use_flash:
            # Flash Attention 2: expand KV for GQA, then use flash_attn_func
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
            else:
                v = v.transpose(1, 2)
            q = q.transpose(1, 2).contiguous()   # (B, S, n_heads, D)
            k = k.transpose(1, 2).contiguous()   # (B, S, n_heads, D)
            v = v.transpose(1, 2).contiguous()   # (B, S, n_heads, D)
            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                causal=True,
            )
        else:
            # Standard scaled dot-product attention (PyTorch 2.0+)
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
            else:
                v = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_out = attn_out.transpose(1, 2)  # (B, S, n_heads, D)

        attn_out = attn_out.reshape(B, S, -1)
        return self.o_proj(attn_out)


# =============================================================================
# Transformer Block (with optional Transformer Engine)
# =============================================================================
class TransformerBlock(nn.Module):
    """Standard transformer block with RMSNorm + SwiGLU FFN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_te = config.use_te

        if self.use_te:
            # Transformer Engine's LayerNormMLP for FP8 acceleration
            self.norm1 = te.RMSNorm(config.d_model)
            self.attn = GQAFlashAttention(config)
            self.norm2 = te.RMSNorm(config.d_model)
            self.ffn = te.LayerNormMLP(
                config.d_model,
                config.d_ff,
                activation="swiglu",
                normalization="RMSNorm",
                bias=config.bias,
            )
        else:
            self.norm1 = RMSNorm(config.d_model)
            self.attn = GQAFlashAttention(config)
            self.norm2 = RMSNorm(config.d_model)
            self.ffn = SwiGLUFFN(config)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        if self.use_te:
            # TE's LayerNormMLP handles norm2 + ffn internally
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.ffn(x))  # TE LayerNormMLP includes norm
        else:
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# GPT Model
# =============================================================================
class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        if config.use_te:
            self.norm_f = te.RMSNorm(config.d_model)
        else:
            self.norm_f = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.drop(self.tok_emb(input_ids))

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def num_parameters(self, non_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n


# =============================================================================
# Dataset
# =============================================================================
class PretrainDataset(Dataset):
    """Simple pre-training dataset from pre-tokenized data."""

    def __init__(self, data_path: str, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        # Memory-mapped for large datasets
        self.data = torch.load(data_path, map_location="cpu", weights_only=True)
        if isinstance(self.data, dict):
            self.data = self.data["input_ids"]
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end].long()
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing / benchmarking."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


# =============================================================================
# Training Loop
# =============================================================================
def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_metrics(step, loss, lr, throughput, elapsed, rank):
    if rank == 0:
        print(
            f"step {step:6d} | loss {loss:.4f} | lr {lr:.2e} | "
            f"{throughput:.0f} tok/s | elapsed {elapsed:.1f}s"
        )
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "train/loss": loss,
                    "train/lr": lr,
                    "train/throughput": throughput,
                    "train/step": step,
                })
        except ImportError:
            pass


def train(args):
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
        print(f"World size: {world_size}")
        attn_backend = "Flash Attention 3" if args.use_flash_attn_3 else f"Flash Attention 2: {args.use_flash_attn}"
        print(f"Attention: {attn_backend}")
        print(f"Transformer Engine (FP8): {args.use_te}")

    # -------------------------------------------------------------------------
    # Model Config (preset sizes)
    # -------------------------------------------------------------------------
    MODEL_CONFIGS = {
        "125m": dict(n_layers=12, n_heads=12, d_model=768),
        "350m": dict(n_layers=24, n_heads=16, d_model=1024),
        "760m": dict(n_layers=24, n_heads=16, d_model=1536),
        "1.3b": dict(n_layers=24, n_heads=32, d_model=2048),
        "2.7b": dict(n_layers=32, n_heads=32, d_model=2560),
        "6.7b": dict(n_layers=32, n_heads=32, d_model=4096),
        "13b":  dict(n_layers=40, n_heads=40, d_model=5120),
    }

    model_kwargs = MODEL_CONFIGS.get(args.model_size, MODEL_CONFIGS["350m"])
    config = ModelConfig(
        max_seq_len=args.seq_len,
        use_flash_attn=args.use_flash_attn,
        use_flash_attn_3=args.use_flash_attn_3,
        use_te=args.use_te,
        dropout=args.dropout,
        **model_kwargs,
    )

    model = GPTModel(config)
    if global_rank == 0:
        n_params = model.num_parameters() / 1e6
        print(f"Model: {args.model_size} ({n_params:.1f}M parameters)")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    if args.data_path and os.path.exists(args.data_path):
        dataset = PretrainDataset(args.data_path, args.seq_len)
    else:
        if global_rank == 0:
            print("Using synthetic data (no --data_path provided)")
        dataset = SyntheticDataset(config.vocab_size, args.seq_len, num_samples=500_000)

    # -------------------------------------------------------------------------
    # Transformer Engine FP8 context
    # -------------------------------------------------------------------------
    fp8_recipe = None
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max",
        )
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
        if global_rank == 0:
            print("FP8 training enabled via Transformer Engine")

    # -------------------------------------------------------------------------
    # DeepSpeed initialization
    # -------------------------------------------------------------------------
    optimizer = None  # Let DeepSpeed create optimizer from config
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        model_parameters=model.parameters(),
    )

    # Manual LR scheduler if DeepSpeed config doesn't define one
    if lr_scheduler is None and optimizer is not None:
        lr_scheduler = get_lr_scheduler(
            optimizer, args.warmup_steps, args.max_steps
        )

    # -------------------------------------------------------------------------
    # Wandb
    # -------------------------------------------------------------------------
    if global_rank == 0 and args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("wandb not installed, skipping logging")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    model_engine.train()
    step = 0
    tokens_processed = 0
    t_start = time.time()
    running_loss = 0.0

    if global_rank == 0:
        print(f"\nStarting training for {args.max_steps} steps...")
        print(f"Batch size per GPU: {model_engine.train_micro_batch_size_per_gpu()}")
        print(f"Gradient accumulation: {model_engine.gradient_accumulation_steps()}")
        print(f"Effective batch size: {model_engine.train_batch_size()}")
        print("-" * 70)

    data_iter = iter(train_loader)

    while step < args.max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        # Forward pass (with optional FP8)
        with fp8_ctx:
            _, loss = model_engine(input_ids, labels)

        # Backward pass
        model_engine.backward(loss)

        # Optimizer step (handles gradient accumulation internally)
        model_engine.step()

        # Update LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Metrics
        running_loss += loss.item()
        tokens_this_step = input_ids.numel() * world_size
        tokens_processed += tokens_this_step
        step += 1

        if step % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t_start
            throughput = tokens_processed / elapsed
            current_lr = optimizer.param_groups[0]["lr"] if optimizer else 0
            log_metrics(step, avg_loss, current_lr, throughput, elapsed, global_rank)
            running_loss = 0.0

        # Checkpointing
        if args.save_interval > 0 and step % args.save_interval == 0:
            if global_rank == 0:
                print(f"Saving checkpoint at step {step}...")
            save_dir = os.path.join(args.save_dir, f"step_{step}")
            model_engine.save_checkpoint(save_dir)

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------
    if global_rank == 0:
        total_time = time.time() - t_start
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Total tokens: {tokens_processed:,}")
        print(f"Average throughput: {tokens_processed / total_time:.0f} tok/s")

    if args.save_dir:
        save_dir = os.path.join(args.save_dir, "final")
        model_engine.save_checkpoint(save_dir)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="GPT Pre-training with DeepSpeed + Flash Attn + TE")

    # Model
    parser.add_argument("--model_size", type=str, default="350m",
                        choices=["125m", "350m", "760m", "1.3b", "2.7b", "6.7b", "13b"])
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Features
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use_te", action="store_true", default=True)
    parser.add_argument("--no_te", action="store_false", dest="use_te")
    parser.add_argument("--use_flash_attn_3", action="store_true", default=False,
                        help="Use Flash Attention 3 via kernels library (requires Hopper GPU)")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enable FP8 training via Transformer Engine (requires Hopper GPU)")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to pre-tokenized .pt file")

    # Training
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default=None)

    # DeepSpeed (added by deepspeed.add_config_arguments)
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
