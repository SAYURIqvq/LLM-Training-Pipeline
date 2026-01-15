"""
Multimodal Training Pipeline (All Stages).

Stage 1: Vision-Language Alignment Pre-training
  - Train projector only (vision encoder + LLM frozen)
  - Image-caption pairs with next-token prediction

Stage 2: Multimodal Instruction Tuning (SFT)
  - Train projector + LLM (vision encoder frozen)
  - Image/video + chat instruction data
  - Loss masking on assistant tokens only

Stage 3: Multimodal RL Post-training
  3a: Multimodal DPO - preference optimization with visual context
  3b: Multimodal GRPO - group-relative policy optimization with visual rewards

Usage:
    # Stage 1: Alignment
    deepspeed --num_gpus=8 multimodal_train.py --stage 1 \
        --deepspeed ds_config_sft.json \
        --data_path data/mm_alignment.jsonl \
        --image_dir data/images/ \
        --max_steps 5000

    # Stage 2: Multimodal SFT
    deepspeed --num_gpus=8 multimodal_train.py --stage 2 \
        --deepspeed ds_config_sft.json \
        --base_model ./checkpoints/mm_stage1_final \
        --data_path data/mm_sft.jsonl \
        --image_dir data/ \
        --max_steps 10000

    # Stage 3a: Multimodal DPO
    deepspeed --num_gpus=8 multimodal_train.py --stage 3 --rl_method dpo \
        --deepspeed ds_config_sft.json \
        --base_model ./checkpoints/mm_stage2_final \
        --data_path data/mm_dpo.jsonl \
        --image_dir data/ \
        --max_steps 2000

    # Stage 3b: Multimodal GRPO
    deepspeed --num_gpus=8 multimodal_train.py --stage 3 --rl_method grpo \
        --deepspeed ds_config_sft.json \
        --base_model ./checkpoints/mm_stage2_final \
        --data_path data/mm_grpo.jsonl \
        --image_dir data/ \
        --group_size 4 --max_steps 1000
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

from train import ModelConfig
from sft import ChatTokenizer
from vision_encoder import VisionConfig
from multimodal_model import MultimodalConfig, MultimodalLLM, create_multimodal_model
from multimodal_data import (
    AlignmentDataset, AlignmentCollator,
    MultimodalSFTDataset, MultimodalSFTCollator,
    MultimodalDPODataset, MultimodalGRPODataset,
)
from dpo import dpo_loss, get_batch_logps
from grpo import (
    GRPOConfig, RuleBasedReward, grpo_advantages, grpo_loss,
    generate_group, compute_ref_log_probs,
)


# =============================================================================
# Stage 1: Alignment Pre-training
# =============================================================================
def train_stage1(args, model_engine, train_loader, fp8_ctx, global_rank):
    """Train projector on image-caption pairs."""
    model_engine.train()
    step = 0
    running_loss = 0.0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"Stage 1: Vision-Language Alignment Pre-training")
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
        pixel_values = batch["pixel_values"].to(model_engine.device)
        image_positions = batch["image_positions"].to(model_engine.device)

        with fp8_ctx:
            _, loss = model_engine(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                image_positions=image_positions,
            )

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t_start
            print(f"step {step:6d} | loss {avg_loss:.4f} | {elapsed:.1f}s")
            running_loss = 0.0

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(
                os.path.join(args.save_dir, f"mm_stage1_step_{step}")
            )

    model_engine.save_checkpoint(os.path.join(args.save_dir, "mm_stage1_final"))
    if global_rank == 0:
        print(f"\nStage 1 complete in {time.time() - t_start:.1f}s")


# =============================================================================
# Stage 2: Multimodal SFT
# =============================================================================
def train_stage2(args, model_engine, train_loader, fp8_ctx, global_rank):
    """Multimodal instruction tuning."""
    model_engine.train()
    step = 0
    running_loss = 0.0
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"Stage 2: Multimodal Instruction Tuning")
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

        # Handle different media types
        pixel_values = batch.get("pixel_values")
        video_frames = batch.get("video_frames")

        if pixel_values is not None:
            pixel_values = pixel_values.to(model_engine.device)
        if video_frames is not None:
            video_frames = video_frames.to(model_engine.device)

        with fp8_ctx:
            _, loss = model_engine(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                video_frames=video_frames,
            )

        model_engine.backward(loss)
        model_engine.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t_start
            print(f"step {step:6d} | loss {avg_loss:.4f} | {elapsed:.1f}s")
            running_loss = 0.0

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(
                os.path.join(args.save_dir, f"mm_stage2_step_{step}")
            )

    model_engine.save_checkpoint(os.path.join(args.save_dir, "mm_stage2_final"))
    if global_rank == 0:
        print(f"\nStage 2 complete in {time.time() - t_start:.1f}s")


# =============================================================================
# Stage 3a: Multimodal DPO
# =============================================================================
def train_stage3_dpo(args, model_engine, ref_model, train_loader, fp8_ctx, global_rank):
    """Multimodal DPO with visual context."""
    model_engine.train()
    step = 0
    running_metrics = {}
    t_start = time.time()

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"Stage 3a: Multimodal DPO (beta={args.beta})")
        print(f"{'='*70}\n")

    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Unpack DPO batch (chosen + rejected concatenated)
        input_ids = batch["input_ids"].to(model_engine.device)
        loss_mask = batch["loss_mask"].to(model_engine.device)
        n_chosen = batch["n_chosen"]
        pixel_values = batch.get("pixel_values")

        if pixel_values is not None:
            pixel_values = pixel_values.to(model_engine.device)
            # Duplicate for chosen + rejected
            pixel_values_full = pixel_values.repeat(2, 1, 1, 1)
        else:
            pixel_values_full = None

        # Policy forward
        with fp8_ctx:
            policy_logits, _ = model_engine(
                input_ids=input_ids,
                pixel_values=pixel_values_full,
            )

        policy_logps = get_batch_logps(policy_logits, input_ids, loss_mask)
        policy_chosen_logps = policy_logps[:n_chosen]
        policy_rejected_logps = policy_logps[n_chosen:]

        # Reference forward
        with torch.no_grad():
            ref_logits, _ = ref_model(
                input_ids=input_ids,
                pixel_values=pixel_values_full,
            )
            ref_logps = get_batch_logps(ref_logits, input_ids, loss_mask)
            ref_chosen_logps = ref_logps[:n_chosen]
            ref_rejected_logps = ref_logps[n_chosen:]

        # DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=args.beta,
        )

        model_engine.backward(loss)
        model_engine.step()

        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0) + v
        step += 1

        if step % args.log_interval == 0 and global_rank == 0:
            n = args.log_interval
            avg = {k: v / n for k, v in running_metrics.items()}
            elapsed = time.time() - t_start
            print(f"step {step:6d} | loss {avg['loss']:.4f} | "
                  f"acc {avg['accuracy']:.3f} | {elapsed:.1f}s")
            running_metrics = {}

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(
                os.path.join(args.save_dir, f"mm_dpo_step_{step}")
            )

    model_engine.save_checkpoint(os.path.join(args.save_dir, "mm_dpo_final"))
    if global_rank == 0:
        print(f"\nMultimodal DPO complete in {time.time() - t_start:.1f}s")


# =============================================================================
# Stage 3b: Multimodal GRPO
# =============================================================================
def train_stage3_grpo(args, model_engine, ref_model, tokenizer, train_loader,
                      fp8_ctx, global_rank):
    """Multimodal GRPO with visual context and rule-based rewards."""
    model_engine.train()
    step = 0
    t_start = time.time()
    reward_history = []

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
    )
    reward_fn = RuleBasedReward()
    kl_coef = grpo_config.kl_coef

    if global_rank == 0:
        print(f"\n{'='*70}")
        print(f"Stage 3b: Multimodal GRPO (G={grpo_config.group_size})")
        print(f"{'='*70}\n")

    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        prompt_ids = batch["input_ids"].to(model_engine.device)
        answers = batch.get("answers", [None] * prompt_ids.shape[0])
        prompt_texts = batch.get("prompt_texts", [""] * prompt_ids.shape[0])

        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(model_engine.device)

        B, prompt_len = prompt_ids.shape
        G = grpo_config.group_size
        policy_model = model_engine.module

        # --- Phase 1: Generate G completions per prompt ---
        with torch.no_grad():
            policy_model.eval()

            # First encode visual features
            visual_tokens = None
            if pixel_values is not None:
                visual_tokens = policy_model.vision(pixel_values=pixel_values)

            # Generate using the LLM part
            # Expand prompts for group
            expanded_ids = prompt_ids.repeat_interleave(G, dim=0)  # (BG, prompt_len)

            # Simple generation (using LLM backbone)
            full_ids, old_log_probs = generate_group(
                policy_model.llm, expanded_ids,
                group_size=1,  # Already expanded
                max_new_tokens=grpo_config.max_new_tokens,
                temperature=grpo_config.temperature,
                top_p=grpo_config.top_p,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
            )

            gen_len = full_ids.shape[1] - prompt_len
            response_ids = full_ids[:, prompt_len:]

            # Decode for reward
            completions = []
            for i in range(B * G):
                tokens = response_ids[i].tolist()
                if tokenizer.eos_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_id)]
                completions.append(tokenizer.decode(tokens))

            expanded_prompts = [p for p in prompt_texts for _ in range(G)]
            expanded_answers = [a for a in answers for _ in range(G)]

            # Rewards
            rewards = reward_fn(expanded_prompts, completions, expanded_answers)
            rewards = rewards.to(model_engine.device).clamp(-5, 5)
            advantages = grpo_advantages(rewards, G)

            # Reference log probs
            ref_lp = compute_ref_log_probs(ref_model.llm, full_ids, prompt_len, gen_len)
            response_mask = (response_ids != tokenizer.pad_id).float()

            policy_model.train()

        # --- Phase 2: GRPO update ---
        BG = full_ids.shape[0]
        perm = torch.randperm(BG, device=full_ids.device)
        mb_size = min(args.grpo_mini_batch, BG)
        total_loss = 0.0
        n_updates = 0

        for start in range(0, BG, mb_size):
            end = min(start + mb_size, BG)
            idx = perm[start:end]

            mb_ids = full_ids[idx]
            mb_old_lp = old_log_probs[idx, :gen_len]
            mb_ref_lp = ref_lp[idx]
            mb_adv = advantages[idx]
            mb_mask = response_mask[idx]

            with fp8_ctx:
                logits, _ = model_engine(mb_ids)
                resp_logits = logits[:, prompt_len - 1:prompt_len - 1 + gen_len, :]
                resp_tokens = mb_ids[:, prompt_len:prompt_len + gen_len]
                new_lp = F.log_softmax(resp_logits, dim=-1)
                new_lp = new_lp.gather(2, resp_tokens.unsqueeze(-1)).squeeze(-1)

                loss, metrics = grpo_loss(
                    new_lp, mb_old_lp, mb_ref_lp, mb_adv, mb_mask,
                    clip_eps=grpo_config.clip_eps,
                    kl_coef=kl_coef,
                )

            model_engine.backward(loss)
            model_engine.step()
            total_loss += loss.item()
            n_updates += 1

        step += 1
        mean_reward = rewards.mean().item()
        reward_history.append(mean_reward)

        if step % args.log_interval == 0 and global_rank == 0:
            elapsed = time.time() - t_start
            recent = sum(reward_history[-args.log_interval:]) / min(len(reward_history), args.log_interval)
            print(f"step {step:5d} | reward {recent:+.3f} | "
                  f"loss {total_loss / max(n_updates, 1):.4f} | {elapsed:.1f}s")

        if args.save_interval > 0 and step % args.save_interval == 0:
            model_engine.save_checkpoint(
                os.path.join(args.save_dir, f"mm_grpo_step_{step}")
            )

    model_engine.save_checkpoint(os.path.join(args.save_dir, "mm_grpo_final"))
    if global_rank == 0:
        print(f"\nMultimodal GRPO complete in {time.time() - t_start:.1f}s")


# =============================================================================
# DPO Collator
# =============================================================================
class MultimodalDPOCollator:
    def __init__(self, pad_id: int, max_len: int, image_size: int = 448):
        self.pad_id = pad_id
        self.max_len = max_len
        self.image_size = image_size

    def __call__(self, batch):
        all_ids = []
        all_masks = []
        pixel_values = []

        max_len = min(
            max(max(len(b["chosen_ids"]) for b in batch),
                max(len(b["rejected_ids"]) for b in batch)),
            self.max_len,
        )

        for key_ids, key_mask in [("chosen_ids", "chosen_mask"), ("rejected_ids", "rejected_mask")]:
            for b in batch:
                ids = b[key_ids][:max_len]
                mask = b[key_mask][:max_len]
                pad_len = max_len - len(ids)
                all_ids.append(F.pad(ids, (0, pad_len), value=self.pad_id))
                all_masks.append(F.pad(mask, (0, pad_len), value=0.0))

        for b in batch:
            pixel_values.append(b["pixel_values"])

        B = len(batch)
        result = {
            "input_ids": torch.stack(all_ids),
            "loss_mask": torch.stack(all_masks),
            "n_chosen": B,
        }
        if pixel_values[0] is not None:
            result["pixel_values"] = torch.stack(pixel_values)

        return result


class MultimodalGRPOCollator:
    def __init__(self, pad_id: int, max_len: int, image_size: int = 448):
        self.pad_id = pad_id
        self.max_len = max_len
        self.image_size = image_size

    def __call__(self, batch):
        ids_list = [b["input_ids"][:self.max_len] for b in batch]
        max_len = max(len(ids) for ids in ids_list)

        padded = [F.pad(ids, (0, max_len - len(ids)), value=self.pad_id) for ids in ids_list]
        answers = [b.get("answer") for b in batch]
        prompts = [b.get("prompt_text", "") for b in batch]

        result = {
            "input_ids": torch.stack(padded),
            "answers": answers,
            "prompt_texts": prompts,
        }

        pixel_values = [b.get("pixel_values") for b in batch]
        if pixel_values[0] is not None:
            result["pixel_values"] = torch.stack([
                pv if pv is not None else torch.zeros(3, self.image_size, self.image_size)
                for pv in pixel_values
            ])

        return result


# =============================================================================
# Main
# =============================================================================
def main(args):
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    model = create_multimodal_model(
        model_size=args.model_size,
        vision_encoder=args.vision_encoder,
        image_size=args.image_size,
        stage=args.stage,
    )

    # Load checkpoint
    if args.base_model and os.path.exists(args.base_model):
        ckpt = os.path.join(args.base_model, "pytorch_model.bin")
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=False)
            if global_rank == 0:
                print(f"Loaded checkpoint from {ckpt}")

    # -------------------------------------------------------------------------
    # Dataset & Collator
    # -------------------------------------------------------------------------
    tokenizer = ChatTokenizer(encoding_name="gpt2")

    if args.stage == 1:
        dataset = AlignmentDataset(
            args.data_path, args.image_dir, tokenizer,
            image_size=args.image_size, max_seq_len=args.seq_len,
        )
        collator = AlignmentCollator(tokenizer.pad_id, args.seq_len)

    elif args.stage == 2:
        dataset = MultimodalSFTDataset(
            args.data_path, args.image_dir, tokenizer,
            image_size=args.image_size, max_seq_len=args.seq_len,
        )
        collator = MultimodalSFTCollator(
            tokenizer.pad_id, args.seq_len, args.image_size,
        )

    elif args.stage == 3 and args.rl_method == "dpo":
        dataset = MultimodalDPODataset(
            args.data_path, args.image_dir, tokenizer,
            image_size=args.image_size, max_seq_len=args.seq_len,
        )
        collator = MultimodalDPOCollator(
            tokenizer.pad_id, args.seq_len, args.image_size,
        )

    elif args.stage == 3 and args.rl_method == "grpo":
        dataset = MultimodalGRPODataset(
            args.data_path, args.image_dir, tokenizer,
            image_size=args.image_size, max_prompt_len=args.max_prompt_len,
        )
        collator = MultimodalGRPOCollator(
            tokenizer.pad_id, args.max_prompt_len, args.image_size,
        )

    # -------------------------------------------------------------------------
    # FP8
    # -------------------------------------------------------------------------
    fp8_ctx = nullcontext()
    if args.fp8:
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max",
        )
        fp8_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    # -------------------------------------------------------------------------
    # DeepSpeed init
    # -------------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        collate_fn=collator,
        model_parameters=trainable_params,
    )

    # -------------------------------------------------------------------------
    # Reference model (for Stage 3)
    # -------------------------------------------------------------------------
    ref_model = None
    if args.stage == 3:
        ref_model = create_multimodal_model(
            model_size=args.model_size,
            vision_encoder=args.vision_encoder,
            image_size=args.image_size,
            stage=3,
        )
        if args.base_model and os.path.exists(args.base_model):
            ckpt = os.path.join(args.base_model, "pytorch_model.bin")
            if os.path.exists(ckpt):
                sd = torch.load(ckpt, map_location="cpu", weights_only=True)
                ref_model.load_state_dict(sd, strict=False)
        ref_model = ref_model.to(f"cuda:{local_rank}")
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    if args.stage == 1:
        train_stage1(args, model_engine, train_loader, fp8_ctx, global_rank)
    elif args.stage == 2:
        train_stage2(args, model_engine, train_loader, fp8_ctx, global_rank)
    elif args.stage == 3 and args.rl_method == "dpo":
        train_stage3_dpo(args, model_engine, ref_model, train_loader, fp8_ctx, global_rank)
    elif args.stage == 3 and args.rl_method == "grpo":
        train_stage3_grpo(
            args, model_engine, ref_model, tokenizer,
            train_loader, fp8_ctx, global_rank,
        )

    if global_rank == 0:
        print("\nTraining complete!")


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal LLM Training Pipeline")

    # Stage
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--rl_method", type=str, default="dpo", choices=["dpo", "grpo"])

    # Model
    parser.add_argument("--model_size", type=str, default="350m")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--vision_encoder", type=str, default=None,
                        help="Pretrained vision encoder (timm model name)")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--fp8", action="store_true", default=False)

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--max_prompt_len", type=int, default=512)

    # DPO
    parser.add_argument("--beta", type=float, default=0.1)

    # GRPO
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--grpo_mini_batch", type=int, default=4)

    # Training
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
