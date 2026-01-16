# LLM Training Pipeline

A complete, from-scratch LLM training pipeline covering **pre-training**, **supervised fine-tuning (SFT)**, **alignment** (DPO / PPO / GRPO), and **multimodal (vision-language) training**.

Built with Flash Attention 2/3, DeepSpeed ZeRO-2/3, and NVIDIA Transformer Engine for FP8 mixed-precision training.

```
                          ┌──→ Reward Model ──→ PPO (RLHF)
                          │    reward_model.py   ppo.py
Pre-training ──→ SFT ──┬─┤
   train.py      sft.py│ ├──→ DPO (offline preference)
                       │ │    dpo.py
                       │ └──→ GRPO (no critic, DeepSeek-R1 style)
                       │      grpo.py
                       │
                       └──→ Multimodal (Vision-Language)
                             Stage 1: Alignment ──→ Stage 2: MM-SFT ──→ Stage 3: MM-DPO / MM-GRPO
                             multimodal_train.py
```

## Features

- **Flash Attention 2/3** — fast and memory-efficient attention (FA3 via `kernels` library for Hopper GPUs)
- **DeepSpeed ZeRO-2/3** — distributed training across multiple GPUs
- **Transformer Engine FP8** — 2x throughput on H100 GPUs
- **LoRA** — memory-efficient fine-tuning with merge support
- **Loss masking** — SFT trains only on assistant tokens
- **Multi-turn conversations** — full chat support in all stages
- **Multiple alignment methods** — DPO, IPO, SimPO, PPO, GRPO
- **GRPO** — DeepSeek-R1 style training with rule-based rewards, no critic needed
- **Adaptive KL penalty** — automatic KL coefficient adjustment for PPO/GRPO
- **Multimodal (VLM)** — LLaVA/InternVL-style vision-language model with 3-stage training
- **Vision Encoder** — ViT with SigLIP/CLIP support, pixel shuffle token compression, dynamic resolution tiling
- **Video understanding** — frame sampling and temporal processing for video inputs

## Architecture

| Component | Implementation |
|-----------|---------------|
| Attention | Flash Attention 2/3 + Grouped-Query Attention (GQA) |
| Position | Rotary Positional Embedding (RoPE) |
| FFN | SwiGLU |
| Norm | RMSNorm |
| Mixed Precision | BF16 / FP8 via Transformer Engine |
| Distributed | DeepSpeed ZeRO-2/3 |
| Fine-tuning | Full parameters / LoRA |
| Vision | ViT / SigLIP / CLIP + Pixel Shuffle + MLP Projector |
| Multimodal | Interleaved image/video-text, dynamic resolution tiling |

### Model Sizes

| Name | Layers | Heads | d_model | Params |
|------|--------|-------|---------|--------|
| 125M | 12 | 12 | 768 | ~125M |
| 350M | 24 | 16 | 1024 | ~350M |
| 1.3B | 24 | 32 | 2048 | ~1.3B |
| 6.7B | 32 | 32 | 4096 | ~6.7B |
| 13B | 40 | 40 | 5120 | ~13B |

## Getting Started

### Installation

```bash
pip install torch>=2.1 deepspeed transformer-engine flash-attn \
    datasets tiktoken wandb tqdm

# Optional: Flash Attention 3 (Hopper GPUs only)
pip install kernels
```

### Quick Test with Synthetic Data

```bash
# Generate sample data for all stages
python generate_sample_data.py --output_dir data/

# Pre-training (single GPU, small model)
deepspeed --num_gpus=1 train.py --model_size 125m --deepspeed ds_config.json --max_steps 100

# SFT
deepspeed --num_gpus=1 sft.py --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --max_steps 100
```

## Usage

### 1. Pre-training

```bash
# Prepare data from HuggingFace
python prepare_data.py --dataset openwebtext --tokenizer gpt2 --output data/owt.pt

# Train
deepspeed --num_gpus=8 train.py \
    --model_size 350m \
    --deepspeed ds_config.json \
    --data_path data/owt.pt \
    --max_steps 50000

# With Flash Attention 3 on Hopper GPUs (1.5-2x faster)
deepspeed --num_gpus=8 train.py \
    --model_size 350m \
    --deepspeed ds_config.json \
    --data_path data/owt.pt \
    --use_flash_attn_3 \
    --max_steps 50000
```

### 2. Supervised Fine-Tuning (SFT)

```bash
# Full fine-tuning
deepspeed --num_gpus=8 sft.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --max_steps 3000

# LoRA fine-tuning (memory-efficient)
deepspeed --num_gpus=8 sft.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --use_lora --lora_rank 16 --lora_alpha 32 \
    --merge_lora_on_save \
    --max_steps 3000
```

### 3a. DPO (Recommended for General Alignment)

```bash
# Standard DPO
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type dpo --beta 0.1 \
    --max_steps 2000

# IPO variant
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type ipo --beta 0.1

# SimPO (reference-free)
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type simpo --beta 2.0
```

### 3b. PPO (Full RLHF)

```bash
# Step 1: Train reward model
deepspeed --num_gpus=8 reward_model.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --max_steps 2000

# Step 2: PPO training
deepspeed --num_gpus=8 ppo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --reward_model ./checkpoints/rm_final \
    --data_path data/prompts.jsonl \
    --max_steps 1000 \
    --ppo_epochs 4 --kl_coef 0.05
```

### 3c. GRPO (Recommended for Reasoning Tasks)

```bash
# Generate math/reasoning data
python generate_grpo_data.py --output data/grpo_prompts.jsonl --num 5000

# Rule-based reward (no reward model needed)
deepspeed --num_gpus=8 grpo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --data_path data/grpo_prompts.jsonl \
    --group_size 8 \
    --kl_coef 0.04 \
    --max_steps 1000

# With reward model
deepspeed --num_gpus=8 grpo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --data_path data/grpo_prompts.jsonl \
    --reward_type model \
    --reward_model_path ./checkpoints/rm_final \
    --group_size 8
```

### 4. Multimodal (Vision-Language) Training

Three-stage training pipeline following the LLaVA/InternVL approach:

```bash
# Generate synthetic multimodal data for testing
python multimodal_data.py

# Stage 1: Vision-Language Alignment (train projector only)
deepspeed --num_gpus=8 multimodal_train.py --stage 1 \
    --deepspeed ds_config_sft.json \
    --data_path data/mm_alignment.jsonl \
    --image_dir data/images/ \
    --max_steps 5000

# Stage 2: Multimodal Instruction Tuning (train projector + LLM)
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

# With pretrained SigLIP vision encoder (recommended)
deepspeed --num_gpus=8 multimodal_train.py --stage 1 \
    --deepspeed ds_config_sft.json \
    --vision_encoder vit_so400m_patch14_siglip_384 \
    --data_path data/mm_alignment.jsonl \
    --image_dir data/images/
```

#### Multimodal Training Stages

| Stage | Trainable | Frozen | Purpose |
|-------|-----------|--------|---------|
| 1 - Alignment | Projector | Vision encoder + LLM | Learn vision-text mapping |
| 2 - Instruction Tuning | Projector + LLM | Vision encoder | Multimodal chat ability |
| 3 - RL Post-training | All | — | Preference alignment (DPO/GRPO) |

## Data Formats

### SFT (`sft_train.jsonl`)

```json
{"messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "Machine learning is..."}
]}
```

### Preference — DPO / Reward Model (`preference_train.jsonl`)

```json
{"chosen": [
    {"role": "user", "content": "Explain AI"},
    {"role": "assistant", "content": "AI is a broad field..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain AI"},
    {"role": "assistant", "content": "idk lol"}
]}
```

### Prompts — PPO (`prompts.jsonl`)

```json
{"prompt": [{"role": "user", "content": "Write a poem about space"}]}
```

### GRPO Prompts (`grpo_prompts.jsonl`)

```json
{"prompt": "What is 25 × 4 + 3?", "answer": "103"}
{"prompt": "Solve for x: 3x + 1 = 10", "answer": "3"}
{"prompt": "Explain how gravity works."}
```

### Multimodal Alignment (`mm_alignment.jsonl`)

```json
{"image": "path/to/image.jpg", "caption": "A cat sitting on a mat"}
```

### Multimodal SFT (`mm_sft.jsonl`)

```json
{"image": "path.jpg", "messages": [
    {"role": "user", "content": "<image>\nDescribe this image."},
    {"role": "assistant", "content": "The image shows..."}
]}
{"video": "path.mp4", "messages": [
    {"role": "user", "content": "<video>\nWhat happens in this video?"},
    {"role": "assistant", "content": "In this video..."}
]}
{"images": ["img1.jpg", "img2.jpg"], "messages": [
    {"role": "user", "content": "<image> <image>\nCompare these two images."},
    {"role": "assistant", "content": "The first image shows..."}
]}
```

### Multimodal DPO (`mm_dpo.jsonl`)

```json
{"image": "path.jpg",
 "chosen": [
    {"role": "user", "content": "<image>\nDescribe"},
    {"role": "assistant", "content": "Detailed description..."}
 ],
 "rejected": [
    {"role": "user", "content": "<image>\nDescribe"},
    {"role": "assistant", "content": "I see an image."}
 ]}
```

### Multimodal GRPO (`mm_grpo.jsonl`)

```json
{"image": "chart.png", "prompt": "What is the highest value?", "answer": "42"}
```

## Alignment Methods Comparison

| | DPO | IPO | SimPO | PPO | GRPO |
|---|---|---|---|---|---|
| Reward model | No | No | No | Yes | Optional |
| Reference model | Yes | Yes | No | Yes | Yes |
| Critic/Value model | No | No | No | Yes | No |
| Online generation | No | No | No | Yes | Yes |
| Memory efficiency | Good | Good | Best | Poor | Good |
| Complexity | Low | Low | Low | High | Medium |
| Best for | General | General | Simple | Complex | Reasoning |

**Recommendations:**
- **DPO** — Simplest. Start here for general alignment with preference data.
- **GRPO** — Best for reasoning/math tasks. No critic needed, rule-based rewards work.
- **PPO** — Most flexible, but complex. Use when you need fine-grained reward shaping.

## DeepSpeed Configs

| Config | ZeRO Stage | Learning Rate | Use For |
|--------|-----------|--------------|---------|
| `ds_config.json` | 2 | 3e-4 | Pre-training (≤2.7B) |
| `ds_config_zero3.json` | 3 | 1.5e-4 | Pre-training (6.7B+) |
| `ds_config_sft.json` | 2 | 2e-5 | SFT / DPO / RM / PPO / GRPO |

## Project Structure

```
├── train.py                 # Pre-training
├── prepare_data.py          # HuggingFace dataset tokenization
├── sft.py                   # Supervised Fine-Tuning + LoRA
├── dpo.py                   # DPO / IPO / SimPO
├── reward_model.py          # Reward model training
├── ppo.py                   # PPO (RLHF)
├── grpo.py                  # GRPO (DeepSeek-R1 style)
├── vision_encoder.py        # ViT encoder + pixel shuffle + projector
├── multimodal_model.py      # Vision-language model (LLaVA/InternVL style)
├── multimodal_train.py      # Multimodal training (all stages)
├── multimodal_data.py       # Multimodal datasets + synthetic data generator
├── generate_sample_data.py  # Synthetic SFT/DPO/PPO test data
├── generate_grpo_data.py    # Math/reasoning GRPO data generator
├── ds_config.json           # DeepSpeed ZeRO-2 config (pre-training)
├── ds_config_zero3.json     # DeepSpeed ZeRO-3 config (large models)
├── ds_config_sft.json       # DeepSpeed config (post-training)
└── LICENSE
```

## References

- [Flash Attention 2](https://arxiv.org/abs/2307.08691) — Dao (2023)
- [Flash Attention 3](https://arxiv.org/abs/2407.08608) — Shah et al. (2024)
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) — Rajbhandari et al. (2020)
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al. (2021)
- [DPO](https://arxiv.org/abs/2305.18290) — Rafailov et al. (2023)
- [IPO](https://arxiv.org/abs/2310.12036) — Azar et al. (2023)
- [SimPO](https://arxiv.org/abs/2405.14734) — Meng et al. (2024)
- [PPO / RLHF](https://arxiv.org/abs/2203.02155) — Ouyang et al. (2022)
- [GRPO / DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek (2025)
- [LLaVA](https://arxiv.org/abs/2304.08485) — Liu et al. (2023)
- [InternVL](https://arxiv.org/abs/2312.14238) — Chen et al. (2023)
- [SigLIP](https://arxiv.org/abs/2303.15343) — Zhai et al. (2023)

## License

[MIT](LICENSE)
