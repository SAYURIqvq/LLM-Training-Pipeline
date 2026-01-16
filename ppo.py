"""
PPO (Proximal Policy Optimization) for RLHF.

Implements the full RLHF-PPO pipeline:
  1. Generate responses from the policy model
  2. Score responses with the reward model
  3. Compute advantages with GAE
  4. Update policy with clipped PPO objective + KL penalty

Usage:
    deepspeed --num_gpus=8 ppo.py --deepspeed ds_config.json \
        --policy_model ./checkpoints/sft_final \
        --reward_model ./checkpoints/rm_final \
        --data_path data/prompts.jsonl

Data format (JSONL):
    {"prompt": [{"role": "user", "content": "Explain quantum computing"}]}
    {"prompt": [{"role": "user", "content": "Write a poem about space"}]}
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset, DataLoader

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from train import GPTModel, ModelConfig
from sft import ChatTokenizer
from reward_model import RewardModel


# =============================================================================
# PPO Config
# =============================================================================
@dataclass
class PPOConfig:
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # PPO
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    vf_coef: float = 0.1
    entropy_coef: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95             # GAE lambda
    kl_coef: float = 0.05         # KL penalty coefficient
    kl_target: float = 6.0        # Target KL for adaptive coefficient
    kl_horizon: int = 10000
    max_grad_norm: float = 1.0

    # Mini-batch
    rollout_batch_size: int = 64
    ppo_mini_batch_size: int = 16

    # Reward
    reward_clip: float = 10.0
    whiten_rewards: bool = True


# =============================================================================
# Generation utilities
# =============================================================================
@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    eos_token_id: int = None,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressive generation with top-k/top-p sampling.
    Returns generated token IDs and per-token log probs.
    """
    model.eval()
    B, prompt_len = input_ids.shape
    device = input_ids.device

    generated_ids = input_ids.clone()
    log_probs = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(generated_ids)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
            next_logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cumprobs - sorted_logits.softmax(dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        token_log_prob = F.log_softmax(next_logits, dim=-1).gather(1, next_token)  # (B, 1)

        # Mask finished sequences
        next_token = next_token.squeeze(-1)
        token_log_prob = token_log_prob.squeeze(-1)
        next_token[finished] = pad_token_id
        token_log_prob[finished] = 0.0

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=1)
        log_probs.append(token_log_prob)

        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

    log_probs = torch.stack(log_probs, dim=1)  # (B, gen_len)
    model.train()
    return generated_ids, log_probs


@torch.no_grad()
def compute_log_probs(model: nn.Module, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Compute per-token log probs for the response portion."""
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for actual tokens (shifted by 1)
    response_logits = log_probs[:, response_start - 1:-1, :]  # predict tokens at response_start onwards
    response_tokens = input_ids[:, response_start:]
    token_log_probs = response_logits.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

    return token_log_probs


# =============================================================================
# GAE (Generalized Advantage Estimation)
# =============================================================================
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns.
    rewards: (B, T) per-token rewards
    values:  (B, T) value estimates
    """
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae

    returns = advantages + values
    return advantages, returns


# =============================================================================
# Value Head
# =============================================================================
class PolicyWithValueHead(nn.Module):
    """Policy model with an additional value head for PPO."""

    def __init__(self, policy: GPTModel):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(policy.config.d_model, policy.config.d_model),
            nn.Tanh(),
            nn.Linear(policy.config.d_model, 1),
        )
        # Zero-init value head
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, input_ids: torch.Tensor, labels=None):
        return self.policy(input_ids, labels)

    def get_values(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get value estimates for each position."""
        x = self.policy.drop(self.policy.tok_emb(input_ids))
        for layer in self.policy.layers:
            x = layer(x)
        x = self.policy.norm_f(x)
        values = self.value_head(x).squeeze(-1)  # (B, S)
        return values


# =============================================================================
# Prompt Dataset
# =============================================================================
class PromptDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: ChatTokenizer, max_prompt_len: int = 512):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.prompts = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data.get("prompt", [])
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                if prompt:
                    self.prompts.append(prompt)

        print(f"Loaded {len(self.prompts)} prompts from {data_path}")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        messages = self.prompts[idx]
        encoded = self.tokenizer.encode_chat(messages)
        ids = encoded["input_ids"][:self.max_prompt_len]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


# =============================================================================
# PPO Trainer
# =============================================================================
class PPOTrainer:
    def __init__(
        self,
        policy_engine,
        ref_model: nn.Module,
        reward_model: nn.Module,
        tokenizer: ChatTokenizer,
        ppo_config: PPOConfig,
        fp8_ctx=None,
    ):
        self.policy_engine = policy_engine
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = ppo_config
        self.fp8_ctx = fp8_ctx or nullcontext()

        self.kl_coef = ppo_config.kl_coef

    @torch.no_grad()
    def generate_rollouts(self, prompt_ids: torch.Tensor) -> Dict:
        """Generate responses and compute rewards."""
        device = prompt_ids.device
        prompt_len = prompt_ids.shape[1]
        policy_model = self.policy_engine.module

        # 1. Generate from policy
        full_ids, gen_log_probs = generate(
            policy_model,
            prompt_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            eos_token_id=self.tokenizer.eos_id,
        )

        response_ids = full_ids[:, prompt_len:]
        gen_len = response_ids.shape[1]

        # 2. Compute reference log probs (for KL penalty)
        ref_log_probs = compute_log_probs(self.ref_model, full_ids, prompt_len)
        ref_log_probs = ref_log_probs[:, :gen_len]
        gen_log_probs = gen_log_probs[:, :gen_len]

        # 3. Compute rewards
        rewards_raw = self.reward_model(full_ids)  # (B,) scalar per sequence
        rewards_raw = rewards_raw.clamp(-self.config.reward_clip, self.config.reward_clip)

        # Distribute reward to last token
        per_token_rewards = torch.zeros(prompt_ids.shape[0], gen_len, device=device)
        per_token_rewards[:, -1] = rewards_raw

        # 4. KL penalty per token
        kl_div = gen_log_probs - ref_log_probs
        per_token_rewards = per_token_rewards - self.kl_coef * kl_div

        # 5. Value estimates
        values = policy_model.get_values(full_ids)[:, prompt_len:prompt_len + gen_len]

        # 6. GAE
        advantages, returns = compute_gae(
            per_token_rewards, values,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )

        # Whiten advantages
        if self.config.whiten_rewards:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "full_ids": full_ids,
            "response_ids": response_ids,
            "old_log_probs": gen_log_probs,
            "ref_log_probs": ref_log_probs,
            "rewards": per_token_rewards,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "prompt_len": prompt_len,
            "raw_rewards": rewards_raw,
            "kl": kl_div.sum(dim=1).mean(),
        }

    def ppo_step(self, rollouts: Dict) -> Dict:
        """Run PPO update epochs on the rollout data."""
        full_ids = rollouts["full_ids"]
        old_log_probs = rollouts["old_log_probs"]
        advantages = rollouts["advantages"]
        returns = rollouts["returns"]
        prompt_len = rollouts["prompt_len"]
        gen_len = rollouts["response_ids"].shape[1]

        B = full_ids.shape[0]
        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.ppo_epochs):
            # Shuffle
            perm = torch.randperm(B, device=full_ids.device)
            for start in range(0, B, self.config.ppo_mini_batch_size):
                end = min(start + self.config.ppo_mini_batch_size, B)
                idx = perm[start:end]

                mb_ids = full_ids[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                with self.fp8_ctx:
                    # New log probs
                    logits, _ = self.policy_engine(mb_ids)
                    new_log_probs = F.log_softmax(logits, dim=-1)
                    response_logprobs = new_log_probs[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
                    response_tokens = mb_ids[:, prompt_len:prompt_len + gen_len]
                    new_lp = response_logprobs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

                    # New values
                    new_values = self.policy_engine.module.get_values(mb_ids)
                    new_values = new_values[:, prompt_len:prompt_len + gen_len]

                    # Ratio
                    ratio = (new_lp - mb_old_lp).exp()

                    # Clipped surrogate objective
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * ratio.clamp(1 - self.config.clip_eps, 1 + self.config.clip_eps)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    vf_loss = 0.5 * (new_values - mb_ret).pow(2).mean()

                    # Entropy bonus
                    probs = F.softmax(logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :], dim=-1)
                    entropy = -(probs * probs.log().clamp(min=-100)).sum(-1).mean()

                    loss = pg_loss + self.config.vf_coef * vf_loss - self.config.entropy_coef * entropy

                self.policy_engine.backward(loss)
                self.policy_engine.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "loss": total_loss / max(n_updates, 1),
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def update_kl_coef(self, kl: float):
        """Adaptive KL penalty coefficient."""
        if kl > self.config.kl_target * 1.5:
            self.kl_coef *= 1.5
        elif kl < self.config.kl_target / 1.5:
            self.kl_coef /= 1.5
        self.kl_coef = max(0.001, min(self.kl_coef, 1.0))


# =============================================================================
# Collator
# =============================================================================
class PromptCollator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        ids_list = [b["input_ids"][:self.max_len] for b in batch]
        max_len = max(len(ids) for ids in ids_list)
        padded = [F.pad(ids, (0, max_len - len(ids)), value=self.pad_id) for ids in ids_list]
        return {"input_ids": torch.stack(padded)}


# =============================================================================
# Main Training
# =============================================================================
def train_ppo(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    ppo_config = PPOConfig(
        max_new_tokens=args.max_new_tokens,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
    )

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
        **model_kwargs,
    )

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    # Policy model (with value head)
    policy_backbone = GPTModel(config)
    policy_model = PolicyWithValueHead(policy_backbone)

    # Reference model (frozen SFT model)
    ref_model = GPTModel(config)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Reward model
    rm_backbone = GPTModel(config)
    reward_model = RewardModel(rm_backbone)
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # Load weights
    def load_weights(model, path, name="model"):
        if path and os.path.exists(path):
            ckpt = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(ckpt):
                sd = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(sd, strict=False)
                if global_rank == 0:
                    print(f"Loaded {name} from {ckpt}")

    load_weights(policy_backbone, args.policy_model, "policy")
    load_weights(ref_model, args.policy_model, "reference")  # Same as SFT
    load_weights(reward_model, args.reward_model, "reward model")

    ref_model = ref_model.to(f"cuda:{local_rank}")
    reward_model = reward_model.to(f"cuda:{local_rank}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    tokenizer = ChatTokenizer(encoding_name="gpt2")
    dataset = PromptDataset(args.data_path, tokenizer, max_prompt_len=args.max_prompt_len)
    collator = PromptCollator(pad_id=tokenizer.pad_id, max_len=args.max_prompt_len)

    # FP8
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # DeepSpeed init (policy only)
    model_engine, optimizer, prompt_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=policy_model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=policy_model.parameters(),
    )

    # -------------------------------------------------------------------------
    # PPO Trainer
    # -------------------------------------------------------------------------
    trainer = PPOTrainer(
        policy_engine=model_engine,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        ppo_config=ppo_config,
        fp8_ctx=fp8_ctx,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    step = 0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"PPO Training | Model: {args.model_size}")
        print(f"Prompts: {len(dataset)} | Max steps: {args.max_steps}")
        print(f"PPO epochs: {ppo_config.ppo_epochs} | KL coef: {ppo_config.kl_coef}")
        print(f"{'='*70}\n")

    data_iter = iter(prompt_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(prompt_loader)
            batch = next(data_iter)

        prompt_ids = batch["input_ids"].to(model_engine.device)

        # 1. Generate rollouts
        rollouts = trainer.generate_rollouts(prompt_ids)

        # 2. PPO update
        ppo_metrics = trainer.ppo_step(rollouts)

        # 3. Adaptive KL
        kl_val = rollouts["kl"].item()
        trainer.update_kl_coef(kl_val)

        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            elapsed = time.time() - t_start
            avg_reward = rollouts["raw_rewards"].mean().item()
            print(
                f"step {step:5d} | reward {avg_reward:+.3f} | "
                f"kl {kl_val:.3f} | kl_coef {trainer.kl_coef:.4f} | "
                f"pg_loss {ppo_metrics['pg_loss']:.4f} | "
                f"vf_loss {ppo_metrics['vf_loss']:.4f} | "
                f"entropy {ppo_metrics['entropy']:.3f} | "
                f"{elapsed:.1f}s"
            )

            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        "ppo/reward": avg_reward,
                        "ppo/kl": kl_val,
                        "ppo/kl_coef": trainer.kl_coef,
                        "ppo/pg_loss": ppo_metrics["pg_loss"],
                        "ppo/vf_loss": ppo_metrics["vf_loss"],
                        "ppo/entropy": ppo_metrics["entropy"],
                        "ppo/step": step,
                    })
            except ImportError:
                pass

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(os.path.join(args.save_dir, f"ppo_step_{step}"))

    # Save
    if args.save_dir:
        model_engine.save_checkpoint(os.path.join(args.save_dir, "ppo_final"))

    if global_rank == 0:
        print(f"\nPPO training complete in {time.time() - t_start:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="PPO (RLHF) Training")
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--policy_model", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--reward_model", type=str, required=True, help="Path to reward model checkpoint")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use_te", action="store_true", default=True)
    parser.add_argument("--no_te", action="store_false", dest="use_te")
    parser.add_argument("--use_flash_attn_3", action="store_true", default=False,
                        help="Use Flash Attention 3 via kernels library (requires Hopper GPU)")
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
