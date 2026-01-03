"""
Direct Preference Optimization (DPO) Training.

DPO is a simpler alternative to PPO for RLHF that directly optimizes the policy
from preference data without needing a separate reward model.

Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model
           is Secretly a Reward Model" (2023)

Also includes:
  - IPO (Identity Preference Optimization)
  - cDPO (conservative DPO with label smoothing)
  - SimPO (Simple Preference Optimization / reference-free)

Usage:
    deepspeed --num_gpus=8 dpo.py --deepspeed ds_config.json \
        --base_model ./checkpoints/sft_final \
        --data_path data/preference_train.jsonl \
        --loss_type dpo --beta 0.1

Data format (JSONL):
    {"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
     "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from train import GPTModel, ModelConfig
from sft import ChatTokenizer, apply_lora


# =============================================================================
# DPO Loss Functions
# =============================================================================
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "dpo",
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute DPO / IPO / SimPO loss.

    Args:
        policy_chosen_logps:   sum log p_policy(y_w | x)
        policy_rejected_logps: sum log p_policy(y_l | x)
        ref_chosen_logps:      sum log p_ref(y_w | x)
        ref_rejected_logps:    sum log p_ref(y_l | x)
        beta: temperature parameter
        label_smoothing: for conservative DPO (cDPO)
        loss_type: "dpo", "ipo", "simpo"
    """
    # Log-ratios
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = chosen_logratios - rejected_logratios  # (B,)

    if loss_type == "dpo":
        # Standard DPO: -log sigmoid(beta * logits)
        if label_smoothing > 0:
            # Conservative DPO (cDPO)
            loss = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            loss = -F.logsigmoid(beta * logits)

    elif loss_type == "ipo":
        # Identity Preference Optimization
        loss = (logits - 1.0 / (2.0 * beta)).pow(2)

    elif loss_type == "simpo":
        # SimPO: reference-free, uses length-normalized log probs
        # logits here are already policy_chosen - policy_rejected (no ref)
        gamma = 0.5  # margin
        simpo_logits = beta * (policy_chosen_logps - policy_rejected_logps) - gamma
        loss = -F.logsigmoid(simpo_logits)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss = loss.mean()

    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "reward_margin": reward_margin.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "chosen_logps": policy_chosen_logps.mean().item(),
        "rejected_logps": policy_rejected_logps.mean().item(),
    }

    return loss, metrics


# =============================================================================
# Log Probability Computation
# =============================================================================
def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Compute log probabilities of the given labels under the logits.

    logits: (B, S, V)
    labels: (B, S)
    loss_mask: (B, S) - 1 for positions to include, 0 to ignore
    """
    # Shift: logits predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)  # (B, S-1)

    # Mask and sum
    masked_logps = per_token_logps * shift_mask
    sum_logps = masked_logps.sum(dim=1)  # (B,)

    if average_log_prob:
        # Length-normalized (used in SimPO)
        lengths = shift_mask.sum(dim=1).clamp(min=1)
        return sum_logps / lengths

    return sum_logps


# =============================================================================
# Dataset
# =============================================================================
class DPODataset(Dataset):
    """Preference pairs for DPO training."""

    def __init__(self, data_path: str, tokenizer: ChatTokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chosen = data.get("chosen", [])
                rejected = data.get("rejected", [])
                if chosen and rejected:
                    self.pairs.append((chosen, rejected))

        print(f"Loaded {len(self.pairs)} DPO pairs from {data_path}")

    def __len__(self):
        return len(self.pairs)

    def _encode(self, messages):
        encoded = self.tokenizer.encode_chat(messages)
        ids = encoded["input_ids"][:self.max_seq_len]
        mask = encoded["loss_mask"][:self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)

    def __getitem__(self, idx):
        chosen_msgs, rejected_msgs = self.pairs[idx]
        chosen_ids, chosen_mask = self._encode(chosen_msgs)
        rejected_ids, rejected_mask = self._encode(rejected_msgs)
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }


class DPOCollator:
    def __init__(self, pad_id: int, max_seq_len: int):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        all_ids = []
        all_masks = []
        max_len = min(
            max(
                max(len(b["chosen_ids"]) for b in batch),
                max(len(b["rejected_ids"]) for b in batch),
            ),
            self.max_seq_len,
        )

        for key_ids, key_mask in [("chosen_ids", "chosen_mask"), ("rejected_ids", "rejected_mask")]:
            for b in batch:
                ids = b[key_ids][:max_len]
                mask = b[key_mask][:max_len]
                pad_len = max_len - len(ids)
                all_ids.append(F.pad(ids, (0, pad_len), value=self.pad_id))
                all_masks.append(F.pad(mask, (0, pad_len), value=0.0))

        B = len(batch)
        return {
            "input_ids": torch.stack(all_ids),       # (2B, S) - first B chosen, last B rejected
            "loss_mask": torch.stack(all_masks),      # (2B, S)
            "n_chosen": B,
        }


# =============================================================================
# Training
# =============================================================================
def train_dpo(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    # Model config
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

    # -------------------------------------------------------------------------
    # Policy model (trainable)
    # -------------------------------------------------------------------------
    policy_model = GPTModel(config)

    # Load SFT weights
    if args.base_model and os.path.exists(args.base_model):
        ckpt_path = os.path.join(args.base_model, "pytorch_model.bin")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            policy_model.load_state_dict(sd, strict=False)
            if global_rank == 0:
                print(f"Loaded policy from {ckpt_path}")

    # Apply LoRA (optional)
    if args.use_lora:
        target_modules = args.lora_targets.split(",") if args.lora_targets else None
        policy_model = apply_lora(
            policy_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
        )

    # -------------------------------------------------------------------------
    # Reference model (frozen copy of SFT model)
    # -------------------------------------------------------------------------
    ref_model = GPTModel(config)
    if args.base_model and os.path.exists(args.base_model):
        ckpt_path = os.path.join(args.base_model, "pytorch_model.bin")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ref_model.load_state_dict(sd, strict=False)

    ref_model = ref_model.to(f"cuda:{local_rank}")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if global_rank == 0:
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy_model.parameters())
        print(f"Policy: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    tokenizer = ChatTokenizer(encoding_name="gpt2")
    dataset = DPODataset(args.data_path, tokenizer, max_seq_len=args.seq_len)
    collator = DPOCollator(pad_id=tokenizer.pad_id, max_seq_len=args.seq_len)

    # FP8
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # DeepSpeed init
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=policy_model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=[p for p in policy_model.parameters() if p.requires_grad],
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    model_engine.train()
    step = 0
    running_metrics = {}
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"DPO Training | Loss: {args.loss_type} | Beta: {args.beta}")
        print(f"Data: {len(dataset)} pairs | Max steps: {args.max_steps}")
        print(f"LoRA: {args.use_lora}")
        print(f"{'='*70}\n")

    use_avg_logp = (args.loss_type == "simpo")
    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model_engine.device)
        loss_mask = batch["loss_mask"].to(model_engine.device)
        n_chosen = batch["n_chosen"]

        with fp8_ctx:
            # Policy forward (chosen + rejected concatenated)
            policy_logits, _ = model_engine(input_ids)

        policy_logps = get_batch_logps(policy_logits, input_ids, loss_mask, average_log_prob=use_avg_logp)
        policy_chosen_logps = policy_logps[:n_chosen]
        policy_rejected_logps = policy_logps[n_chosen:]

        # Reference forward
        with torch.no_grad():
            ref_logits, _ = ref_model(input_ids)
            ref_logps = get_batch_logps(ref_logits, input_ids, loss_mask, average_log_prob=use_avg_logp)
            ref_chosen_logps = ref_logps[:n_chosen]
            ref_rejected_logps = ref_logps[n_chosen:]

        # DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=args.beta,
            label_smoothing=args.label_smoothing,
            loss_type=args.loss_type,
        )

        model_engine.backward(loss)
        model_engine.step()

        # Accumulate metrics
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0) + v
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            elapsed = time.time() - t_start
            n = args.log_interval
            avg = {k: v / n for k, v in running_metrics.items()}
            print(
                f"step {step:6d} | loss {avg['loss']:.4f} | "
                f"acc {avg['accuracy']:.3f} | margin {avg['reward_margin']:.3f} | "
                f"{elapsed:.1f}s"
            )
            running_metrics = {}

            try:
                import wandb
                if wandb.run:
                    wandb.log({f"dpo/{k}": v for k, v in avg.items()} | {"dpo/step": step})
            except ImportError:
                pass

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(os.path.join(args.save_dir, f"dpo_step_{step}"))

    # Save
    if args.save_dir:
        model_engine.save_checkpoint(os.path.join(args.save_dir, "dpo_final"))

    if global_rank == 0:
        print(f"\nDPO training complete in {time.time() - t_start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="DPO / IPO / SimPO Training")

    # Model
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--base_model", type=str, default=None, help="Path to SFT checkpoint")
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

    # DPO
    parser.add_argument("--loss_type", type=str, default="dpo", choices=["dpo", "ipo", "simpo"])
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="cDPO label smoothing")

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,o_proj")

    # Data
    parser.add_argument("--data_path", type=str, required=True)

    # Training
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
