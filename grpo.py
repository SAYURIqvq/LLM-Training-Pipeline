"""
GRPO (Group Relative Policy Optimization) Training.

GRPO is a reinforcement learning method introduced in DeepSeek-R1 that eliminates
the need for a separate critic/value model by estimating advantages from group-level
relative rewards within a batch of sampled outputs.

Key differences from PPO:
  - No value/critic network needed (saves ~50% memory)
  - Generates G completions per prompt, uses group-level reward normalization
  - Advantages estimated from relative rewards within the group
  - Simpler implementation with comparable or better performance

Reference: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
           Reinforcement Learning (2025)

Also includes:
  - Outcome-based reward (rule-based, no reward model needed)
  - Process reward model support
  - Multi-reward composition (correctness + format + length)

Usage:
    deepspeed --num_gpus=8 grpo.py --deepspeed ds_config_sft.json \
        --policy_model ./checkpoints/sft_final \
        --data_path data/grpo_prompts.jsonl \
        --group_size 8 --max_steps 1000

Data format (JSONL):
    {"prompt": "What is 2+3?", "answer": "5"}
    {"prompt": "Solve: 3x + 1 = 10", "answer": "3"}
    {"prompt": [{"role": "user", "content": "Explain gravity"}]}
"""

import argparse
import json
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset, DataLoader

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from train import GPTModel, ModelConfig
from sft import ChatTokenizer


# =============================================================================
# GRPO Config
# =============================================================================
@dataclass
class GRPOConfig:
    # Group sampling
    group_size: int = 8            # G: number of completions per prompt
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    # GRPO optimization
    grpo_epochs: int = 1           # μ: number of optimization iterations per batch
    mini_batch_size: int = 4       # Mini-batch size within each epoch
    clip_eps: float = 0.2          # ε: PPO-style clipping
    kl_coef: float = 0.04          # β: KL divergence penalty coefficient
    kl_type: str = "kl"            # "kl" (standard) or "abs" (absolute)
    max_grad_norm: float = 1.0
    entropy_coef: float = 0.0      # Optional entropy bonus

    # Reward
    reward_type: str = "rule"      # "rule" (outcome-based) or "model" (reward model)
    reward_clip: float = 5.0
    length_penalty: float = 0.0    # Penalty per token to encourage conciseness

    # Adaptive KL
    kl_target: Optional[float] = None  # If set, adaptively adjust kl_coef


# =============================================================================
# Reward Functions (Rule-Based / Outcome-Based)
# =============================================================================
class RewardFunction:
    """Base class for reward computation."""

    def __call__(self, prompts: List[str], completions: List[str],
                 references: List[Optional[str]]) -> torch.Tensor:
        raise NotImplementedError


class RuleBasedReward(RewardFunction):
    """
    Outcome-based reward for verifiable tasks (math, code, etc.).
    Combines multiple reward signals:
      - Correctness: Does the answer match the reference?
      - Format: Does the output follow expected format (e.g., <think>...</think>)?
      - Length: Penalty for overly long outputs
    """

    def __init__(self, length_penalty: float = 0.0):
        self.length_penalty = length_penalty

    def extract_answer(self, text: str) -> str:
        """Extract final answer from completion (handles various formats)."""
        # Try boxed format: \boxed{answer}
        boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed[-1].strip()

        # Try "The answer is X" format
        answer_match = re.search(
            r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([^\n.]+)',
            text, re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).strip()

        # Try <answer>X</answer> format
        tag_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if tag_match:
            return tag_match.group(1).strip()

        # Fall back to last line / last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        return text.strip().split('\n')[-1].strip()

    def check_format(self, text: str) -> float:
        """Check if output follows expected reasoning format."""
        score = 0.0

        # Reward for showing reasoning steps
        has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
        if has_think:
            score += 0.2

        # Reward for step-by-step markers
        has_steps = bool(re.search(r'(?:step\s+\d|first|then|therefore|finally)', text, re.IGNORECASE))
        if has_steps:
            score += 0.1

        return score

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison."""
        answer = answer.strip().lower()
        # Remove trailing punctuation
        answer = re.sub(r'[.,;:!?]+$', '', answer)
        # Normalize whitespace
        answer = ' '.join(answer.split())
        # Try to parse as number
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return f"{num:.6g}"
        except ValueError:
            return answer

    def __call__(self, prompts: List[str], completions: List[str],
                 references: List[Optional[str]]) -> torch.Tensor:
        rewards = []
        for prompt, completion, reference in zip(prompts, completions, references):
            reward = 0.0

            # Correctness reward
            if reference is not None:
                predicted = self.extract_answer(completion)
                pred_norm = self.normalize_answer(predicted)
                ref_norm = self.normalize_answer(reference)

                if pred_norm == ref_norm:
                    reward += 1.0  # Full credit
                else:
                    # Partial credit for numeric proximity
                    try:
                        pred_val = float(pred_norm)
                        ref_val = float(ref_norm)
                        rel_error = abs(pred_val - ref_val) / max(abs(ref_val), 1e-8)
                        if rel_error < 0.01:
                            reward += 0.8
                        elif rel_error < 0.1:
                            reward += 0.3
                    except ValueError:
                        pass

            # Format reward
            reward += self.check_format(completion)

            # Length penalty
            n_tokens = len(completion.split())
            reward -= self.length_penalty * n_tokens

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


class CompositeReward(RewardFunction):
    """Combine multiple reward functions with weights."""

    def __init__(self, reward_fns: List[Tuple[RewardFunction, float]]):
        self.reward_fns = reward_fns  # [(fn, weight), ...]

    def __call__(self, prompts, completions, references) -> torch.Tensor:
        total = torch.zeros(len(prompts))
        for fn, weight in self.reward_fns:
            total += weight * fn(prompts, completions, references)
        return total


class RewardModelWrapper(RewardFunction):
    """Wraps a trained reward model for use as a reward function."""

    def __init__(self, reward_model: nn.Module, tokenizer: ChatTokenizer,
                 max_len: int = 2048, device: str = "cuda"):
        self.model = reward_model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts, completions, references) -> torch.Tensor:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            encoded = self.tokenizer.encode_chat(messages)
            ids = torch.tensor(encoded["input_ids"][:self.max_len], dtype=torch.long)
            ids = ids.unsqueeze(0).to(self.device)
            reward = self.model(ids)
            rewards.append(reward.item())
        return torch.tensor(rewards, dtype=torch.float32)


# =============================================================================
# Generation
# =============================================================================
@torch.no_grad()
def generate_group(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    eos_token_id: int = None,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate G completions for each prompt.

    Args:
        prompt_ids: (B, prompt_len)
        group_size: G completions per prompt

    Returns:
        full_ids: (B*G, prompt_len + gen_len)
        log_probs: (B*G, gen_len) per-token log probabilities
    """
    model.eval()
    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    # Repeat each prompt G times: (B*G, prompt_len)
    expanded = prompt_ids.repeat_interleave(group_size, dim=0)
    BG = expanded.shape[0]

    generated = expanded.clone()
    all_log_probs = []
    finished = torch.zeros(BG, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        logits, _ = model(generated)
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-k
        if top_k > 0:
            topk_vals = torch.topk(next_logits, top_k)[0][:, -1:]
            next_logits[next_logits < topk_vals] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = (cum_probs - sorted_logits.softmax(dim=-1)) > top_p
            sorted_logits[mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (BG,)
        token_lp = F.log_softmax(next_logits, dim=-1).gather(1, next_token.unsqueeze(1)).squeeze(1)

        next_token[finished] = pad_token_id
        token_lp[finished] = 0.0

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        all_log_probs.append(token_lp)

        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

    log_probs = torch.stack(all_log_probs, dim=1)  # (BG, gen_len)
    model.train()
    return generated, log_probs


@torch.no_grad()
def compute_ref_log_probs(
    ref_model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    gen_len: int,
) -> torch.Tensor:
    """Compute per-token log probs from the reference model."""
    logits, _ = ref_model(full_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # Align: logits at position t predict token at t+1
    response_lp = log_probs[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
    response_tokens = full_ids[:, prompt_len:prompt_len + gen_len]
    per_token_lp = response_lp.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

    return per_token_lp  # (BG, gen_len)


# =============================================================================
# GRPO Core Algorithm
# =============================================================================
def grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Compute GRPO advantages using group-relative normalization.

    For each prompt i with G completions, advantage for completion j is:
        A_ij = (r_ij - mean(r_i)) / std(r_i)

    Args:
        rewards: (B*G,) scalar rewards for each completion
        group_size: G

    Returns:
        advantages: (B*G,) normalized advantages
    """
    B = rewards.shape[0] // group_size

    # Reshape to (B, G) for group-level normalization
    grouped = rewards.view(B, group_size)

    # Group mean and std
    group_mean = grouped.mean(dim=1, keepdim=True)  # (B, 1)
    group_std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)

    # Normalize within group
    advantages = (grouped - group_mean) / group_std  # (B, G)

    return advantages.view(-1)  # (B*G,)


def grpo_loss(
    new_log_probs: torch.Tensor,      # (BG, T) per-token log probs from policy
    old_log_probs: torch.Tensor,      # (BG, T) per-token log probs from old policy
    ref_log_probs: torch.Tensor,      # (BG, T) per-token log probs from reference
    advantages: torch.Tensor,          # (BG,) per-sequence advantages
    response_mask: torch.Tensor,       # (BG, T) mask for response tokens
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
    kl_type: str = "kl",
    entropy_coef: float = 0.0,
    new_logits: Optional[torch.Tensor] = None,  # For entropy computation
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute the GRPO loss.

    L_GRPO = -1/(BG) Σ_i Σ_t [
        min(r_t * A_i, clip(r_t, 1-ε, 1+ε) * A_i)
        - β * KL(π || π_ref)
    ]

    where r_t = π(o_t|q, o_{<t}) / π_old(o_t|q, o_{<t})
    """
    # Per-token importance ratio
    ratio = (new_log_probs - old_log_probs).exp()  # (BG, T)

    # Expand advantages to per-token: (BG,) -> (BG, T)
    adv_expanded = advantages.unsqueeze(1).expand_as(ratio)

    # Clipped surrogate
    pg_loss1 = -adv_expanded * ratio
    pg_loss2 = -adv_expanded * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # KL penalty (per-token)
    if kl_type == "kl":
        # Standard KL: D_KL(π || π_ref) = Σ π(x) [log π(x) - log π_ref(x)]
        # Approximated token-level: log π - log π_ref
        kl_per_token = new_log_probs - ref_log_probs
    elif kl_type == "abs":
        # Absolute difference: |log π - log π_ref|
        kl_per_token = (new_log_probs - ref_log_probs).abs()
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    # Masked mean over response tokens
    pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)
    kl_loss = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)

    loss = pg_loss + kl_coef * kl_loss

    # Optional entropy bonus
    entropy = 0.0
    if entropy_coef > 0 and new_logits is not None:
        probs = F.softmax(new_logits, dim=-1)
        ent = -(probs * probs.log().clamp(min=-100)).sum(-1)
        entropy = (ent * response_mask).sum() / response_mask.sum().clamp(min=1)
        loss = loss - entropy_coef * entropy

    # Metrics
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float()
        clip_frac = (clip_frac * response_mask).sum() / response_mask.sum().clamp(min=1)
        approx_kl = kl_per_token.mean().item()

    metrics = {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "kl": approx_kl,
        "clip_frac": clip_frac.item(),
        "entropy": entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
    }

    return loss, metrics


# =============================================================================
# Prompt Dataset
# =============================================================================
class GRPOPromptDataset(Dataset):
    """Dataset of prompts with optional reference answers."""

    def __init__(self, data_path: str, tokenizer: ChatTokenizer, max_prompt_len: int = 512):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.items = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Handle different formats
                prompt = data.get("prompt", data.get("question", data.get("input", "")))
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
                    prompt = [{"role": "user", "content": prompt[0]}]

                answer = data.get("answer", data.get("reference", data.get("target", None)))
                prompt_text = prompt[-1]["content"] if prompt else ""

                self.items.append({
                    "messages": prompt,
                    "answer": str(answer) if answer is not None else None,
                    "prompt_text": prompt_text,
                })

        print(f"Loaded {len(self.items)} GRPO prompts from {data_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        encoded = self.tokenizer.encode_chat(item["messages"])
        ids = encoded["input_ids"][:self.max_prompt_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "answer": item["answer"],
            "prompt_text": item["prompt_text"],
        }


class GRPOCollator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        ids_list = [b["input_ids"][:self.max_len] for b in batch]
        max_len = max(len(ids) for ids in ids_list)

        padded = [F.pad(ids, (0, max_len - len(ids)), value=self.pad_id) for ids in ids_list]
        answers = [b["answer"] for b in batch]
        prompt_texts = [b["prompt_text"] for b in batch]

        return {
            "input_ids": torch.stack(padded),
            "answers": answers,
            "prompt_texts": prompt_texts,
        }


# =============================================================================
# GRPO Trainer
# =============================================================================
class GRPOTrainer:
    def __init__(
        self,
        policy_engine,
        ref_model: nn.Module,
        tokenizer: ChatTokenizer,
        reward_fn: RewardFunction,
        config: GRPOConfig,
        fp8_ctx=None,
    ):
        self.policy_engine = policy_engine
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.fp8_ctx = fp8_ctx or nullcontext()
        self.kl_coef = config.kl_coef

    @torch.no_grad()
    def generate_and_score(
        self,
        prompt_ids: torch.Tensor,
        prompt_texts: List[str],
        answers: List[Optional[str]],
    ) -> Dict:
        """
        Phase 1: Generate G completions per prompt and compute rewards + advantages.
        """
        device = prompt_ids.device
        B, prompt_len = prompt_ids.shape
        G = self.config.group_size
        policy_model = self.policy_engine.module

        # 1. Generate G completions per prompt
        full_ids, old_log_probs = generate_group(
            policy_model,
            prompt_ids,
            group_size=G,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            eos_token_id=self.tokenizer.eos_id,
            pad_token_id=self.tokenizer.pad_id,
        )

        gen_len = full_ids.shape[1] - prompt_len
        response_ids = full_ids[:, prompt_len:]  # (BG, gen_len)

        # 2. Decode completions for reward computation
        completions = []
        for i in range(B * G):
            resp_tokens = response_ids[i].tolist()
            # Truncate at EOS
            if self.tokenizer.eos_id in resp_tokens:
                eos_idx = resp_tokens.index(self.tokenizer.eos_id)
                resp_tokens = resp_tokens[:eos_idx]
            completions.append(self.tokenizer.decode(resp_tokens))

        # Expand prompts and answers to match BG
        expanded_prompts = [p for p in prompt_texts for _ in range(G)]
        expanded_answers = [a for a in answers for _ in range(G)]

        # 3. Compute rewards
        rewards = self.reward_fn(expanded_prompts, completions, expanded_answers)
        rewards = rewards.to(device)
        rewards = rewards.clamp(-self.config.reward_clip, self.config.reward_clip)

        # 4. Compute group-relative advantages
        advantages = grpo_advantages(rewards, G)

        # 5. Reference model log probs
        ref_log_probs = compute_ref_log_probs(self.ref_model, full_ids, prompt_len, gen_len)

        # 6. Response mask (1 for actual tokens, 0 for padding/post-EOS)
        response_mask = (response_ids != self.tokenizer.pad_id).float()

        return {
            "full_ids": full_ids,           # (BG, prompt_len + gen_len)
            "old_log_probs": old_log_probs[:, :gen_len],  # (BG, gen_len)
            "ref_log_probs": ref_log_probs,  # (BG, gen_len)
            "rewards": rewards,              # (BG,)
            "advantages": advantages,        # (BG,)
            "response_mask": response_mask,  # (BG, gen_len)
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "completions": completions,
        }

    def optimization_step(self, rollouts: Dict) -> Dict:
        """
        Phase 2: GRPO policy optimization.
        Run μ epochs of mini-batch updates on the generated data.
        """
        full_ids = rollouts["full_ids"]
        old_log_probs = rollouts["old_log_probs"]
        ref_log_probs = rollouts["ref_log_probs"]
        advantages = rollouts["advantages"]
        response_mask = rollouts["response_mask"]
        prompt_len = rollouts["prompt_len"]
        gen_len = rollouts["gen_len"]

        BG = full_ids.shape[0]
        total_metrics = {}
        n_updates = 0

        for epoch in range(self.config.grpo_epochs):
            perm = torch.randperm(BG, device=full_ids.device)

            for start in range(0, BG, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, BG)
                idx = perm[start:end]

                mb_ids = full_ids[idx]
                mb_old_lp = old_log_probs[idx]
                mb_ref_lp = ref_log_probs[idx]
                mb_adv = advantages[idx]
                mb_mask = response_mask[idx]

                with self.fp8_ctx:
                    # Forward pass: get new log probs
                    logits, _ = self.policy_engine(mb_ids)

                    # Extract response log probs
                    resp_logits = logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
                    resp_tokens = mb_ids[:, prompt_len:prompt_len + gen_len]
                    new_log_probs = F.log_softmax(resp_logits, dim=-1)
                    new_lp = new_log_probs.gather(2, resp_tokens.unsqueeze(-1)).squeeze(-1)

                    # GRPO loss
                    loss, metrics = grpo_loss(
                        new_log_probs=new_lp,
                        old_log_probs=mb_old_lp,
                        ref_log_probs=mb_ref_lp,
                        advantages=mb_adv,
                        response_mask=mb_mask,
                        clip_eps=self.config.clip_eps,
                        kl_coef=self.kl_coef,
                        kl_type=self.config.kl_type,
                        entropy_coef=self.config.entropy_coef,
                        new_logits=resp_logits if self.config.entropy_coef > 0 else None,
                    )

                self.policy_engine.backward(loss)
                self.policy_engine.step()

                # Accumulate metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                n_updates += 1

        # Average metrics
        avg_metrics = {k: v / max(n_updates, 1) for k, v in total_metrics.items()}
        return avg_metrics

    def update_kl_coef(self, kl: float):
        """Adaptive KL coefficient adjustment."""
        if self.config.kl_target is None:
            return
        if kl > self.config.kl_target * 1.5:
            self.kl_coef = min(self.kl_coef * 1.5, 1.0)
        elif kl < self.config.kl_target / 1.5:
            self.kl_coef = max(self.kl_coef / 1.5, 0.001)


# =============================================================================
# Main Training
# =============================================================================
def train_grpo(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        grpo_epochs=args.grpo_epochs,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        kl_type=args.kl_type,
        mini_batch_size=args.mini_batch_size,
        reward_type=args.reward_type,
        length_penalty=args.length_penalty,
        entropy_coef=args.entropy_coef,
        kl_target=args.kl_target,
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
    # Policy model
    policy_model = GPTModel(config)

    # Reference model (frozen)
    ref_model = GPTModel(config)

    # Load weights
    def load_weights(model, path, name="model"):
        if path and os.path.exists(path):
            ckpt = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(ckpt):
                sd = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(sd, strict=False)
                if global_rank == 0:
                    print(f"Loaded {name} from {ckpt}")

    load_weights(policy_model, args.policy_model, "policy")
    load_weights(ref_model, args.policy_model, "reference")

    ref_model = ref_model.to(f"cuda:{local_rank}")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------------------
    # Reward function
    # -------------------------------------------------------------------------
    if grpo_config.reward_type == "rule":
        reward_fn = RuleBasedReward(length_penalty=grpo_config.length_penalty)
        if global_rank == 0:
            print("Using rule-based (outcome) reward")
    elif grpo_config.reward_type == "model":
        from reward_model import RewardModel
        rm_backbone = GPTModel(config)
        rm = RewardModel(rm_backbone)
        load_weights(rm, args.reward_model_path, "reward model")
        rm = rm.to(f"cuda:{local_rank}")
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False
        tokenizer_for_rm = ChatTokenizer(encoding_name="gpt2")
        reward_fn = RewardModelWrapper(rm, tokenizer_for_rm, device=f"cuda:{local_rank}")
        if global_rank == 0:
            print("Using trained reward model")
    else:
        raise ValueError(f"Unknown reward_type: {grpo_config.reward_type}")

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    tokenizer = ChatTokenizer(encoding_name="gpt2")
    dataset = GRPOPromptDataset(args.data_path, tokenizer, max_prompt_len=args.max_prompt_len)
    collator = GRPOCollator(pad_id=tokenizer.pad_id, max_len=args.max_prompt_len)

    # FP8
    fp8_ctx = nullcontext()
    if args.use_te and args.fp8:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"
        )
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # DeepSpeed init
    model_engine, optimizer, prompt_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=policy_model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=policy_model.parameters(),
    )

    # -------------------------------------------------------------------------
    # GRPO Trainer
    # -------------------------------------------------------------------------
    trainer = GRPOTrainer(
        policy_engine=model_engine,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=grpo_config,
        fp8_ctx=fp8_ctx,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    step = 0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"GRPO Training | Model: {args.model_size}")
        print(f"Prompts: {len(dataset)} | Group size: {grpo_config.group_size}")
        print(f"Max steps: {args.max_steps} | GRPO epochs: {grpo_config.grpo_epochs}")
        print(f"KL coef: {grpo_config.kl_coef} | Clip: {grpo_config.clip_eps}")
        print(f"Reward: {grpo_config.reward_type}")
        print(f"{'='*70}\n")

    data_iter = iter(prompt_loader)
    reward_history = []

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(prompt_loader)
            batch = next(data_iter)

        prompt_ids = batch["input_ids"].to(model_engine.device)
        prompt_texts = batch["prompt_texts"]
        answers = batch["answers"]

        # Phase 1: Generate and score
        rollouts = trainer.generate_and_score(prompt_ids, prompt_texts, answers)

        # Phase 2: GRPO optimization
        metrics = trainer.optimization_step(rollouts)

        # Adaptive KL
        trainer.update_kl_coef(metrics.get("kl", 0))

        step += 1
        mean_reward = rollouts["rewards"].mean().item()
        reward_history.append(mean_reward)

        if step % args.log_interval == 0 and global_rank == 0:
            elapsed = time.time() - t_start
            recent_reward = sum(reward_history[-args.log_interval:]) / min(len(reward_history), args.log_interval)

            # Show a sample completion
            sample_idx = 0
            sample_completion = rollouts["completions"][sample_idx][:200]

            print(
                f"step {step:5d} | "
                f"reward {recent_reward:+.3f} | "
                f"kl {metrics.get('kl', 0):.4f} | "
                f"kl_coef {trainer.kl_coef:.4f} | "
                f"pg_loss {metrics.get('pg_loss', 0):.4f} | "
                f"clip {metrics.get('clip_frac', 0):.3f} | "
                f"{elapsed:.1f}s"
            )
            if step % (args.log_interval * 5) == 0:
                print(f"  Sample: {sample_completion}...")

            try:
                import wandb
                if wandb.run:
                    wandb.log({
                        "grpo/reward": recent_reward,
                        "grpo/kl": metrics.get("kl", 0),
                        "grpo/kl_coef": trainer.kl_coef,
                        "grpo/pg_loss": metrics.get("pg_loss", 0),
                        "grpo/kl_loss": metrics.get("kl_loss", 0),
                        "grpo/clip_frac": metrics.get("clip_frac", 0),
                        "grpo/entropy": metrics.get("entropy", 0),
                        "grpo/loss": metrics.get("loss", 0),
                        "grpo/step": step,
                    })
            except ImportError:
                pass

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(os.path.join(args.save_dir, f"grpo_step_{step}"))

    # Save final
    if args.save_dir:
        model_engine.save_checkpoint(os.path.join(args.save_dir, "grpo_final"))

    if global_rank == 0:
        total_time = time.time() - t_start
        print(f"\nGRPO training complete in {total_time:.1f}s")
        print(f"Final avg reward: {sum(reward_history[-50:]) / min(len(reward_history), 50):.3f}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training (DeepSeek-R1 style)")

    # Model
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--policy_model", type=str, default=None, help="Path to SFT checkpoint")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Features
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", action="store_false", dest="use_flash_attn")
    parser.add_argument("--use_te", action="store_true", default=True)
    parser.add_argument("--no_te", action="store_false", dest="use_te")
    parser.add_argument("--use_flash_attn_3", action="store_true", default=False,
                        help="Use Flash Attention 3 via kernels library (requires Hopper GPU)")
    parser.add_argument("--fp8", action="store_true", default=False)

    # GRPO
    parser.add_argument("--group_size", type=int, default=8, help="G: completions per prompt")
    parser.add_argument("--grpo_epochs", type=int, default=1, help="μ: optimization epochs per batch")
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--kl_type", type=str, default="kl", choices=["kl", "abs"])
    parser.add_argument("--kl_target", type=float, default=None, help="Adaptive KL target")
    parser.add_argument("--entropy_coef", type=float, default=0.0)

    # Reward
    parser.add_argument("--reward_type", type=str, default="rule", choices=["rule", "model"])
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--length_penalty", type=float, default=0.0)

    # Training
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo(args)
