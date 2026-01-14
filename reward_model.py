"""
Reward Model Training for RLHF.

Trains a reward model from human preference data (chosen vs rejected pairs).
The reward model shares the same architecture as the base LLM but replaces the
LM head with a scalar reward head.

Usage:
    deepspeed --num_gpus=8 reward_model.py --deepspeed ds_config.json \
        --base_model ./checkpoints/sft_final \
        --data_path data/preference_train.jsonl

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
from sft import ChatTokenizer


# =============================================================================
# Reward Model
# =============================================================================
class RewardModel(nn.Module):
    """GPT-based reward model: backbone + scalar reward head."""

    def __init__(self, backbone: GPTModel):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.d_model, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Returns scalar reward for each sequence.
        Uses the last non-padding token's representation.
        """
        # Get hidden states from backbone (skip LM head)
        x = self.backbone.drop(self.backbone.tok_emb(input_ids))
        for layer in self.backbone.layers:
            x = layer(x)
        x = self.backbone.norm_f(x)  # (B, S, D)

        # Find last non-padding token
        if attention_mask is not None:
            # Index of last 1 in attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(x.size(0), device=x.device)
            last_hidden = x[batch_idx, seq_lengths]  # (B, D)
        else:
            last_hidden = x[:, -1]  # (B, D)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward


# =============================================================================
# Dataset
# =============================================================================
class PreferenceDataset(Dataset):
    """Preference pairs: chosen vs rejected responses."""

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

        print(f"Loaded {len(self.pairs)} preference pairs from {data_path}")

    def __len__(self):
        return len(self.pairs)

    def _tokenize(self, messages):
        encoded = self.tokenizer.encode_chat(messages)
        ids = encoded["input_ids"][:self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        chosen_msgs, rejected_msgs = self.pairs[idx]
        return {
            "chosen_ids": self._tokenize(chosen_msgs),
            "rejected_ids": self._tokenize(rejected_msgs),
        }


class PreferenceCollator:
    """Pads chosen and rejected sequences, concatenating into a single batch."""

    def __init__(self, pad_id: int, max_seq_len: int):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        all_ids = []
        all_masks = []

        # Process chosen then rejected (interleaved: chosen_0, rejected_0, chosen_1, ...)
        chosen_list = [b["chosen_ids"] for b in batch]
        rejected_list = [b["rejected_ids"] for b in batch]

        max_len = min(
            max(max(len(c) for c in chosen_list), max(len(r) for r in rejected_list)),
            self.max_seq_len,
        )

        for ids_list in [chosen_list, rejected_list]:
            for ids in ids_list:
                ids = ids[:max_len]
                pad_len = max_len - len(ids)
                padded = F.pad(ids, (0, pad_len), value=self.pad_id)
                mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
                all_ids.append(padded)
                all_masks.append(mask)

        B = len(batch)
        input_ids = torch.stack(all_ids)            # (2B, S)
        attention_mask = torch.stack(all_masks).long()  # (2B, S)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_chosen": B,  # First B are chosen, last B are rejected
        }


# =============================================================================
# Reward Model Loss
# =============================================================================
def reward_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Bradley-Terry preference model loss.
    loss = -log(sigmoid(r_chosen - r_rejected))
    """
    diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(diff).mean()

    # Metrics
    accuracy = (diff > 0).float().mean()
    reward_margin = diff.mean()

    return loss, {
        "accuracy": accuracy.item(),
        "reward_margin": reward_margin.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


# =============================================================================
# Training
# =============================================================================
def train_reward_model(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    # Model
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
        **model_kwargs,
    )

    backbone = GPTModel(config)

    # Load SFT weights
    if args.base_model and os.path.exists(args.base_model):
        ckpt_path = os.path.join(args.base_model, "pytorch_model.bin")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict, strict=False)
            if global_rank == 0:
                print(f"Loaded base model from {ckpt_path}")

    model = RewardModel(backbone)

    # Dataset
    tokenizer = ChatTokenizer(encoding_name="gpt2")
    dataset = PreferenceDataset(args.data_path, tokenizer, max_seq_len=args.seq_len)
    collator = PreferenceCollator(pad_id=tokenizer.pad_id, max_seq_len=args.seq_len)

    # FP8
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # DeepSpeed
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=model.parameters(),
    )

    # Train
    model_engine.train()
    step = 0
    running_loss = 0.0
    running_acc = 0.0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"Reward Model Training | {len(dataset)} preference pairs")
        print(f"{'='*70}\n")

    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        n_chosen = batch["n_chosen"]

        with fp8_ctx:
            rewards = model_engine(input_ids, attention_mask)  # (2B,)

        chosen_rewards = rewards[:n_chosen]
        rejected_rewards = rewards[n_chosen:]

        loss, metrics = reward_loss(chosen_rewards, rejected_rewards)

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        running_acc += metrics["accuracy"]
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            avg_loss = running_loss / args.log_interval
            avg_acc = running_acc / args.log_interval
            elapsed = time.time() - t_start
            print(f"step {step:6d} | loss {avg_loss:.4f} | acc {avg_acc:.3f} | {elapsed:.1f}s")
            running_loss = 0.0
            running_acc = 0.0

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(os.path.join(args.save_dir, f"rm_step_{step}"))

    # Save
    if args.save_dir:
        model_engine.save_checkpoint(os.path.join(args.save_dir, "rm_final"))

    if global_rank == 0:
        print(f"\nReward model training complete in {time.time() - t_start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Reward Model Training")
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use_te", action="store_true", default=True)
    parser.add_argument("--no_te", action="store_false", dest="use_te")
    parser.add_argument("--use_flash_attn_3", action="store_true", default=False,
                        help="Use Flash Attention 3 via kernels library (requires Hopper GPU)")
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_reward_model(args)
