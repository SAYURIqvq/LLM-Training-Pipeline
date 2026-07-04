# LLM-Training-Pipeline

一个完整的、从零开始构建的大语言模型（LLM）训练流水线，涵盖**预训练（Pre-training）**、**监督微调（SFT）**、**对齐（DPO / PPO / GRPO）**以及**多模态（视觉-语言）训练**。

基于 Flash Attention 2/3、DeepSpeed ZeRO-2/3 和 NVIDIA Transformer Engine，实现 FP8 混合精度训练与高效分布式训练。

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

## 特性

- **Flash Attention 2/3** —— 高效且低显存的注意力机制（在 Hopper GPU 上通过 `kernels` 库支持 FA3）

- **DeepSpeed ZeRO-2/3** —— 支持多 GPU 分布式训练

- **Transformer Engine FP8** —— 在 H100 GPU 上实现约 2 倍吞吐提升

- **LoRA** —— 高效低显存微调，并支持权重合并

- **Loss Masking** —— SFT 仅在 assistant token 上计算损失

- **多轮对话支持** —— 全流程支持多轮对话训练

- **多种对齐方法** —— 支持 DPO、IPO、SimPO、PPO、GRPO

- **GRPO** —— 类 DeepSeek-R1 的训练方式，基于规则奖励，无需 critic

- **自适应 KL 惩罚** —— 自动调整 PPO / GRPO 中的 KL 系数

- **多模态（VLM）** —— 类 LLaVA / InternVL 的三阶段视觉语言模型训练

- **视觉编码器** —— 基于 ViT，支持 SigLIP / CLIP，包含 Pixel Shuffle token 压缩与动态分辨率切片

- **视频理解能力** —— 支持视频帧采样与时序建模处理

## 模型架构

| 组件 | 实现方式 |
|------|----------|
| 注意力机制 | Flash Attention 2/3 + 分组查询注意力（GQA） |
| 位置编码 | 旋转位置编码（RoPE） |
| 前馈网络（FFN） | SwiGLU |
| 归一化 | RMSNorm |
| 混合精度 | 基于 Transformer Engine 的 BF16 / FP8 |
| 分布式训练 | DeepSpeed ZeRO-2/3 |
| 微调方式 | 全参数微调 / LoRA |
| 视觉模块 | ViT / SigLIP / CLIP + Pixel Shuffle + MLP 投影层 |
| 多模态融合 | 图像/视频与文本交错建模，支持动态分辨率切片 |

---

### 模型规模

| 模型名称 | 层数 | 注意力头数 | 隐藏维度（d_model） | 参数量 |
|----------|------|------------|---------------------|--------|
| 125M | 12 | 12 | 768 | 约 125M |
| 350M | 24 | 16 | 1024 | 约 350M |
| 1.3B | 24 | 32 | 2048 | 约 1.3B |
| 6.7B | 32 | 32 | 4096 | 约 6.7B |
| 13B | 40 | 40 | 5120 | 约 13B |

## 快速开始

### 1. 数据生成、配置校验与本地 smoke test

下面这组命令不需要 GPU，可用于检查样例数据格式、DeepSpeed 配置结构和轻量校验逻辑：

```bash
python generate_sample_data.py --output_dir data/smoke --num_sft 8 --num_preference 4 --num_prompts 4 --seed 42

python pipeline_validation.py \
  --sft data/smoke/sft_train.jsonl \
  --preference data/smoke/preference_train.jsonl \
  --prompts data/smoke/prompts.jsonl \
  --deepspeed ds_config.json \
  --deepspeed ds_config_sft.json \
  --deepspeed ds_config_zero3.json

python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

### 2. 监督微调（SFT）

```bash
# 全参数微调
deepspeed --num_gpus=8 sft.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/final \
--data_path data/sft_train.jsonl \
--max_steps 3000

# LoRA 微调（更省显存）
deepspeed --num_gpus=8 sft.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/final \
--data_path data/sft_train.jsonl \
--use_lora --lora_rank 16 --lora_alpha 32 \
--merge_lora_on_save \
--max_steps 3000
```

---

### 3a. DPO（推荐用于通用对齐）

```bash
# 标准 DPO
deepspeed --num_gpus=8 dpo.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/sft_final \
--data_path data/preference_train.jsonl \
--loss_type dpo --beta 0.1 \
--max_steps 2000

# IPO 变体
deepspeed --num_gpus=8 dpo.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/sft_final \
--data_path data/preference_train.jsonl \
--loss_type ipo --beta 0.1

# SimPO（无需参考模型）
deepspeed --num_gpus=8 dpo.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/sft_final \
--data_path data/preference_train.jsonl \
--loss_type simpo --beta 2.0
```

---

### 3b. PPO（完整 RLHF 流程）

```bash
# 步骤1：训练奖励模型（Reward Model）
deepspeed --num_gpus=8 reward_model.py \
--deepspeed ds_config_sft.json \
--base_model ./checkpoints/sft_final \
--data_path data/preference_train.jsonl \
--max_steps 2000

# 步骤2：进行 PPO 训练
deepspeed --num_gpus=8 ppo.py \
--deepspeed ds_config_sft.json \
--policy_model ./checkpoints/sft_final \
--reward_model ./checkpoints/rm_final \
--data_path data/prompts.jsonl \
--max_steps 1000 \
--ppo_epochs 4 --kl_coef 0.05
```
### 3c. GRPO（推荐用于推理任务）

```json
{"image": "chart.png", "prompt": "最高的数值是多少？", "answer": "42"}
```

---

## 对齐方法对比

| | DPO | IPO | SimPO | PPO | GRPO |
|---|---|---|---|---|---|
| 是否需要奖励模型 | 否 | 否 | 否 | 是 | 可选 |
| 是否需要参考模型 | 是 | 是 | 否 | 是 | 是 |
| 是否需要 Critic / Value 模型 | 否 | 否 | 否 | 是 | 否 |
| 是否在线生成 | 否 | 否 | 否 | 是 | 是 |
| 显存效率 | 较好 | 较好 | 最优 | 较差 | 较好 |
| 实现复杂度 | 低 | 低 | 低 | 高 | 中 |
| 适用场景 | 通用 | 通用 | 简单任务 | 复杂任务 | 推理任务 |

---

### 使用建议

- **DPO** —— 最简单的方法。推荐用于基于偏好数据的通用对齐任务  
- **GRPO** —— 最适合推理 / 数学任务。不需要 critic，可使用规则奖励  
- **PPO** —— 最灵活但实现复杂。适用于需要精细奖励设计的场景  

---

## DeepSpeed 配置说明

| 配置文件 | ZeRO 阶段 | 学习率 | 适用场景 |
|----------|-----------|--------|----------|
| `ds_config.json` | 2 | 3e-4 | 预训练（≤2.7B） |
| `ds_config_zero3.json` | 3 | 1.5e-4 | 预训练（6.7B+） |
| `ds_config_sft.json` | 2 | 2e-5 | SFT / DPO / RM / PPO / GRPO |

---

## 项目结构

```
├── train.py                 # 预训练
├── prepare_data.py          # HuggingFace 数据集处理与分词
├── sft.py                   # 监督微调（SFT）+ LoRA
├── dpo.py                   # DPO / IPO / SimPO
├── reward_model.py          # 奖励模型训练
├── ppo.py                   # PPO（RLHF）
├── grpo.py                  # GRPO（DeepSeek-R1 风格）
├── vision_encoder.py        # ViT 编码器 + Pixel Shuffle + 投影层
├── multimodal_model.py      # 视觉语言模型（LLaVA / InternVL 风格）
├── multimodal_train.py      # 多模态训练（全流程）
├── multimodal_data.py       # 多模态数据集与数据生成
├── generate_sample_data.py  # SFT / DPO / PPO 测试数据生成
├── generate_grpo_data.py    # 推理 / 数学任务 GRPO 数据生成
├── ds_config.json           # DeepSpeed ZeRO-2 配置（预训练）
├── ds_config_zero3.json     # DeepSpeed ZeRO-3 配置（大模型）
├── ds_config_sft.json       # DeepSpeed 配置（后训练阶段）
└── LICENSE
```
## 参考

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
