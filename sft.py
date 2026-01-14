"""
Supervised Fine-Tuning (SFT) with DeepSpeed + Flash Attention + Transformer Engine.

Supports:
  - Chat/instruction format with proper loss masking (only train on assistant tokens)
  - LoRA / full fine-tuning
  - Packing multiple examples into a single sequence
  - Multi-turn conversation support

Usage:
    deepspeed --num_gpus=8 sft.py --deepspeed ds_config.json \
        --base_model ./checkpoints/final \
        --data_path data/sft_train.jsonl \
        --max_steps 3000

Data format (JSONL):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset, DataLoader

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Import model from pre-training
from train import GPTModel, ModelConfig


# =============================================================================
# LoRA
# =============================================================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Init: A = normal, B = zero → initial output is zero
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_b(self.lora_a(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the original linear layer."""
        with torch.no_grad():
            merged_weight = self.original.weight + (
                self.lora_b.weight @ self.lora_a.weight
            ) * self.scaling
            self.original.weight.copy_(merged_weight)
        return self.original


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: List[str] = None,
) -> nn.Module:
    """Apply LoRA to specified linear layers in the model."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                lora_count += 1

    print(f"Applied LoRA to {lora_count} layers (rank={rank}, alpha={alpha})")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA layers back into the base model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = dict(model.named_modules())[parent_name]
            merged = module.merge()
            setattr(parent, attr_name, merged)
    return model


# =============================================================================
# Chat Tokenizer Wrapper
# =============================================================================
class ChatTokenizer:
    """Wraps tiktoken with chat template formatting."""

    # Special tokens
    BOS = "<|begin_of_text|>"
    EOS = "<|end_of_text|>"
    SYSTEM_START = "<|start_header_id|>system<|end_header_id|>\n\n"
    USER_START = "<|start_header_id|>user<|end_header_id|>\n\n"
    ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    TURN_END = "<|eot_id|>"

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

        # Reserve special token IDs (use high-range IDs)
        self.bos_id = self.vocab_size
        self.eos_id = self.vocab_size + 1
        self.pad_id = self.vocab_size + 2

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: List[int]) -> str:
        # Filter out special tokens
        ids = [i for i in ids if i < self.vocab_size]
        return self.enc.decode(ids)

    def encode_chat(self, messages: List[Dict]) -> Dict[str, List[int]]:
        """
        Encode a chat conversation, returning input_ids and a loss mask.
        Loss mask = 1 for assistant tokens (tokens we train on), 0 for others.
        """
        input_ids = [self.bos_id]
        loss_mask = [0]  # Don't train on BOS

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                header_tokens = self.encode(self.SYSTEM_START)
            elif role == "user":
                header_tokens = self.encode(self.USER_START)
            elif role == "assistant":
                header_tokens = self.encode(self.ASSISTANT_START)
            else:
                continue

            content_tokens = self.encode(content)
            turn_end_tokens = self.encode(self.TURN_END)

            # Header: don't train
            input_ids.extend(header_tokens)
            loss_mask.extend([0] * len(header_tokens))

            # Content
            input_ids.extend(content_tokens)
            if role == "assistant":
                loss_mask.extend([1] * len(content_tokens))  # Train on assistant
            else:
                loss_mask.extend([0] * len(content_tokens))  # Don't train on user/system

            # Turn end
            input_ids.extend(turn_end_tokens)
            if role == "assistant":
                loss_mask.extend([1] * len(turn_end_tokens))  # Include EoT in training
            else:
                loss_mask.extend([0] * len(turn_end_tokens))

        input_ids.append(self.eos_id)
        loss_mask.append(1)  # Train on EOS

        return {"input_ids": input_ids, "loss_mask": loss_mask}


# =============================================================================
# SFT Dataset
# =============================================================================
class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning from JSONL chat data."""

    def __init__(self, data_path: str, tokenizer: ChatTokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                messages = data.get("messages") or data.get("conversations", [])
                if messages:
                    self.examples.append(messages)

        print(f"Loaded {len(self.examples)} SFT examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]
        encoded = self.tokenizer.encode_chat(messages)

        input_ids = encoded["input_ids"][:self.max_seq_len]
        loss_mask = encoded["loss_mask"][:self.max_seq_len]

        # Labels: shift by 1 for next-token prediction
        labels = input_ids[1:] + [self.tokenizer.pad_id]
        loss_mask = loss_mask[1:] + [0]

        # Apply loss mask: set non-trainable positions to -100
        labels = [l if m == 1 else -100 for l, m in zip(labels, loss_mask)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTCollator:
    """Pads batch to max length with right-padding."""

    def __init__(self, pad_id: int, max_seq_len: int):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        max_len = min(max(len(b["input_ids"]) for b in batch), self.max_seq_len)

        input_ids = []
        labels = []
        attention_mask = []

        for b in batch:
            ids = b["input_ids"][:max_len]
            lab = b["labels"][:max_len]
            pad_len = max_len - len(ids)

            input_ids.append(F.pad(ids, (0, pad_len), value=self.pad_id))
            labels.append(F.pad(lab, (0, pad_len), value=-100))
            mask = torch.ones(len(ids), dtype=torch.long)
            attention_mask.append(F.pad(mask, (0, pad_len), value=0))

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


# =============================================================================
# Training
# =============================================================================
def train_sft(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    # -------------------------------------------------------------------------
    # Load base model
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

    # Load pre-trained weights
    if args.base_model and os.path.exists(args.base_model):
        if global_rank == 0:
            print(f"Loading base model from {args.base_model}")
        # DeepSpeed checkpoint
        _, client_state = model.load_checkpoint(args.base_model) if hasattr(model, "load_checkpoint") else (None, None)
        if client_state is None:
            # Try loading as a plain state_dict
            ckpt_path = os.path.join(args.base_model, "pytorch_model.bin")
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
            else:
                # Try DeepSpeed format
                try:
                    model.load_state_dict(
                        torch.load(
                            os.path.join(args.base_model, "mp_rank_00_model_states.pt"),
                            map_location="cpu",
                            weights_only=True,
                        )["module"],
                        strict=False,
                    )
                except Exception as e:
                    if global_rank == 0:
                        print(f"Warning: Could not load checkpoint: {e}")
                        print("Starting from random initialization.")

    # -------------------------------------------------------------------------
    # Apply LoRA (optional)
    # -------------------------------------------------------------------------
    if args.use_lora:
        target_modules = args.lora_targets.split(",") if args.lora_targets else None
        model = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if global_rank == 0:
            print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    # -------------------------------------------------------------------------
    # Dataset & Collator
    # -------------------------------------------------------------------------
    tokenizer = ChatTokenizer(encoding_name="gpt2")
    dataset = SFTDataset(args.data_path, tokenizer, max_seq_len=args.seq_len)
    collator = SFTCollator(pad_id=tokenizer.pad_id, max_seq_len=args.seq_len)

    # -------------------------------------------------------------------------
    # FP8 context
    # -------------------------------------------------------------------------
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # -------------------------------------------------------------------------
    # DeepSpeed init
    # -------------------------------------------------------------------------
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    model_engine.train()
    step = 0
    running_loss = 0.0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"SFT Training | Model: {args.model_size} | LoRA: {args.use_lora}")
        print(f"Data: {len(dataset)} examples | Max steps: {args.max_steps}")
        print(f"Batch/GPU: {model_engine.train_micro_batch_size_per_gpu()} | "
              f"Grad accum: {model_engine.gradient_accumulation_steps()}")
        print(f"{'='*70}\n")

    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        with fp8_ctx:
            _, loss = model_engine(input_ids, labels)

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t_start
            lr = optimizer.param_groups[0]["lr"] if optimizer else 0
            print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            running_loss = 0.0

            try:
                import wandb
                if wandb.run:
                    wandb.log({"sft/loss": avg_loss, "sft/lr": lr, "sft/step": step})
            except ImportError:
                pass

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_dir = os.path.join(args.save_dir, f"sft_step_{step}")
            model_engine.save_checkpoint(save_dir)

    # -------------------------------------------------------------------------
    # Save final
    # -------------------------------------------------------------------------
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, "sft_final")
        model_engine.save_checkpoint(save_dir)

        # Optionally merge LoRA and save
        if args.use_lora and args.merge_lora_on_save:
            if global_rank == 0:
                print("Merging LoRA weights...")
                merged_model = merge_lora(model)
                merged_path = os.path.join(args.save_dir, "sft_merged")
                os.makedirs(merged_path, exist_ok=True)
                torch.save(merged_model.state_dict(), os.path.join(merged_path, "pytorch_model.bin"))
                print(f"Merged model saved to {merged_path}")

    if global_rank == 0:
        print(f"\nSFT training complete in {time.time() - t_start:.1f}s")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SFT with DeepSpeed + Flash Attn + TE")

    # Model
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--base_model", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Features
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use_te", action="store_true", default=True)
    parser.add_argument("--no_te", action="store_false", dest="use_te")
    parser.add_argument("--use_flash_attn_3", action="store_true", default=False,
                        help="Use Flash Attention 3 via kernels library (requires Hopper GPU)")
    parser.add_argument("--fp8", action="store_true", default=False)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated target module names")
    parser.add_argument("--merge_lora_on_save", action="store_true", default=False)

    # Data
    parser.add_argument("--data_path", type=str, required=True)

    # Training
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
