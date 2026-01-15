"""
Multimodal LLM (Vision-Language Model).

Architecture: ViT-MLP-LLM (LLaVA / InternVL style)
  - Vision Encoder (ViT/SigLIP) extracts image/video features
  - MLP Projector aligns vision→text embedding space
  - GPT backbone processes interleaved visual + text tokens

Supports:
  - Single image, multi-image, and video inputs
  - Interleaved image-text conversations
  - Dynamic resolution via tiling
  - Selective freezing for staged training
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import GPTModel, ModelConfig
from vision_encoder import VisionConfig, VisionModule


# =============================================================================
# Multimodal Config
# =============================================================================
@dataclass
class MultimodalConfig:
    # LLM
    llm_config: ModelConfig = None
    # Vision
    vision_config: VisionConfig = None
    # Pretrained encoder (None = train from scratch)
    pretrained_vision_encoder: str = None
    # Special token IDs
    image_token_id: int = 50300     # <image> placeholder token
    video_token_id: int = 50301     # <video> placeholder token
    # Freeze settings for staged training
    freeze_vision: bool = True
    freeze_llm: bool = False
    freeze_projector: bool = False

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = ModelConfig()
        if self.vision_config is None:
            self.vision_config = VisionConfig(
                llm_hidden_size=self.llm_config.d_model,
            )
        # Ensure alignment
        self.vision_config.llm_hidden_size = self.llm_config.d_model


# =============================================================================
# Multimodal LLM
# =============================================================================
class MultimodalLLM(nn.Module):
    """
    Full multimodal model: Vision Encoder + Projector + LLM.

    During forward pass:
      1. Encode images/videos → visual tokens
      2. Replace <image>/<video> placeholders in text with visual tokens
      3. Run combined sequence through the LLM
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # LLM backbone
        self.llm = GPTModel(config.llm_config)

        # Vision module (encoder + projector)
        self.vision = VisionModule(
            config.vision_config,
            pretrained_encoder=config.pretrained_vision_encoder,
        )

        # Apply freeze settings
        self._apply_freeze(config)

    def _apply_freeze(self, config: MultimodalConfig):
        """Freeze/unfreeze components for staged training."""
        if config.freeze_vision:
            for p in self.vision.encoder.parameters():
                p.requires_grad = False
            print("Vision encoder: frozen")

        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            print("LLM backbone: frozen")

        if config.freeze_projector:
            for p in self.vision.projector.parameters():
                p.requires_grad = False
            print("Projector: frozen")

        # Always keep special tokens trainable
        self.vision.image_start_token.requires_grad = True
        self.vision.image_end_token.requires_grad = True
        self.vision.frame_separator.requires_grad = True

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
              f"({100 * trainable / total:.2f}%)")

    def set_stage(self, stage: int):
        """
        Configure freezing for different training stages.

        Stage 1: Alignment pre-training (freeze vision + LLM, train projector)
        Stage 2: Instruction tuning (freeze vision, train projector + LLM)
        Stage 3: Full fine-tuning or RL (everything trainable)
        """
        if stage == 1:
            # Alignment: only train projector
            for p in self.vision.encoder.parameters():
                p.requires_grad = False
            for p in self.llm.parameters():
                p.requires_grad = False
            for p in self.vision.projector.parameters():
                p.requires_grad = True
            self.vision.image_start_token.requires_grad = True
            self.vision.image_end_token.requires_grad = True
            self.vision.frame_separator.requires_grad = True
            if hasattr(self.vision, 'pixel_shuffle') and self.vision.pixel_shuffle:
                for p in self.vision.pixel_shuffle.parameters():
                    p.requires_grad = True
            print("Stage 1: Training projector only")

        elif stage == 2:
            # Instruction tuning: train projector + LLM
            for p in self.vision.encoder.parameters():
                p.requires_grad = False
            for p in self.llm.parameters():
                p.requires_grad = True
            for p in self.vision.projector.parameters():
                p.requires_grad = True
            print("Stage 2: Training projector + LLM")

        elif stage == 3:
            # Full fine-tuning
            for p in self.parameters():
                p.requires_grad = True
            print("Stage 3: Full fine-tuning (all parameters)")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")

    def _merge_visual_tokens(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace <image> token embeddings with visual tokens.

        Args:
            input_ids: (B, S)
            text_embeds: (B, S, D) text token embeddings
            visual_tokens: (B, num_visual_tokens, D)
            image_positions: (B,) position of <image> token in each sequence

        Returns:
            merged_embeds: (B, S + num_visual - 1, D)
        """
        B, S, D = text_embeds.shape
        num_visual = visual_tokens.shape[1]

        merged_list = []
        for b in range(B):
            pos = image_positions[b].item()
            if pos < 0 or pos >= S:
                # No image token, use text only
                merged_list.append(text_embeds[b])
                continue

            # Before image token
            before = text_embeds[b, :pos]  # (pos, D)
            # After image token (skip the <image> placeholder)
            after = text_embeds[b, pos + 1:]  # (S - pos - 1, D)
            # Insert visual tokens
            vis = visual_tokens[b]  # (num_visual, D)

            merged = torch.cat([before, vis, after], dim=0)
            merged_list.append(merged)

        # Pad to same length
        max_len = max(m.shape[0] for m in merged_list)
        padded = torch.zeros(B, max_len, D, device=text_embeds.device, dtype=text_embeds.dtype)
        for b, m in enumerate(merged_list):
            padded[b, :m.shape[0]] = m

        return padded

    def _merge_multi_image_tokens(
        self,
        text_embeds: torch.Tensor,
        image_token_id: int,
        input_ids: torch.Tensor,
        all_visual_tokens: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Handle multiple images in a single sequence.
        Replaces each <image> token with corresponding visual tokens.
        """
        B, S, D = text_embeds.shape

        merged_list = []
        for b in range(B):
            # Find all image token positions
            image_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            num_images = min(len(image_positions), len(all_visual_tokens))

            if num_images == 0:
                merged_list.append(text_embeds[b])
                continue

            segments = []
            prev_pos = 0
            for img_idx in range(num_images):
                pos = image_positions[img_idx].item()
                # Text before this image token
                segments.append(text_embeds[b, prev_pos:pos])
                # Visual tokens for this image
                if img_idx < len(all_visual_tokens):
                    vis = all_visual_tokens[img_idx]
                    if vis.dim() == 3:
                        vis = vis[b] if vis.shape[0] > b else vis[0]
                    segments.append(vis)
                prev_pos = pos + 1

            # Remaining text after last image
            segments.append(text_embeds[b, prev_pos:])
            merged_list.append(torch.cat(segments, dim=0))

        max_len = max(m.shape[0] for m in merged_list)
        padded = torch.zeros(B, max_len, D, device=text_embeds.device, dtype=text_embeds.dtype)
        for b, m in enumerate(merged_list):
            padded[b, :m.shape[0]] = m

        return padded

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
        multi_images: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with interleaved text and visual inputs.

        Args:
            input_ids: (B, S) token IDs with <image>/<video> placeholders
            labels: (B, S) for loss computation (-100 = ignore)
            pixel_values: (B, C, H, W) single image per sample
            video_frames: (B, T, C, H, W) video frames
            image_positions: (B,) position of <image> token
            multi_images: List of (C, H, W) tensors for multi-image

        Returns:
            logits: (B, S', V) output logits
            loss: scalar loss (if labels provided)
        """
        B = input_ids.shape[0]

        # 1. Get text embeddings
        text_embeds = self.llm.tok_emb(input_ids)  # (B, S, D)

        # 2. Encode visual inputs
        visual_tokens = None
        if pixel_values is not None:
            visual_tokens = self.vision(pixel_values=pixel_values)  # (B, N, D)
        elif video_frames is not None:
            visual_tokens = self.vision(video_frames=video_frames)  # (B, N, D)

        # 3. Merge visual tokens into text sequence
        if visual_tokens is not None and image_positions is not None:
            merged_embeds = self._merge_visual_tokens(
                input_ids, text_embeds, visual_tokens, image_positions,
            )
        elif visual_tokens is not None:
            # Auto-find <image> token positions
            merged_embeds = self._merge_multi_image_tokens(
                text_embeds, self.config.image_token_id, input_ids,
                [visual_tokens],
            )
        elif multi_images is not None:
            # Multiple images
            all_vis = []
            for img in multi_images:
                vis = self.vision(pixel_values=img.unsqueeze(0))
                all_vis.append(vis)
            merged_embeds = self._merge_multi_image_tokens(
                text_embeds, self.config.image_token_id, input_ids, all_vis,
            )
        else:
            merged_embeds = text_embeds

        # 4. Forward through LLM layers
        x = self.llm.drop(merged_embeds)
        for layer in self.llm.layers:
            x = layer(x)
        x = self.llm.norm_f(x)
        logits = self.llm.lm_head(x)

        # 5. Compute loss
        loss = None
        if labels is not None:
            # Align labels with merged sequence length
            S_out = logits.shape[1]
            if labels.shape[1] < S_out:
                # Pad labels with -100
                pad_len = S_out - labels.shape[1]
                labels = F.pad(labels, (0, pad_len), value=-100)
            elif labels.shape[1] > S_out:
                labels = labels[:, :S_out]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Generate text conditioned on image/video + text prompt."""
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        # Encode visual once
        visual_tokens = None
        if pixel_values is not None:
            visual_tokens = self.vision(pixel_values=pixel_values)
        elif video_frames is not None:
            visual_tokens = self.vision(video_frames=video_frames)

        # Build initial embeddings
        text_embeds = self.llm.tok_emb(input_ids)
        if visual_tokens is not None:
            merged = self._merge_multi_image_tokens(
                text_embeds, self.config.image_token_id, input_ids,
                [visual_tokens],
            )
        else:
            merged = text_embeds

        # Autoregressive generation
        generated = input_ids.clone()
        current_embeds = merged

        for _ in range(max_new_tokens):
            x = self.llm.drop(current_embeds)
            for layer in self.llm.layers:
                x = layer(x)
            x = self.llm.norm_f(x)
            logits = self.llm.lm_head(x)

            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = (cum_probs - sorted_logits.softmax(dim=-1)) > top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Embed new token and append
            new_embed = self.llm.tok_emb(next_token)
            current_embeds = torch.cat([current_embeds, new_embed], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        self.train()
        return generated

    def num_parameters(self):
        """Report parameter counts by component."""
        vision_params = sum(p.numel() for p in self.vision.encoder.parameters())
        projector_params = sum(p.numel() for p in self.vision.projector.parameters())
        llm_params = sum(p.numel() for p in self.llm.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "vision_encoder": vision_params,
            "projector": projector_params,
            "llm": llm_params,
            "total": total,
        }


# =============================================================================
# Model Factory
# =============================================================================
def create_multimodal_model(
    model_size: str = "350m",
    vision_encoder: str = None,
    image_size: int = 448,
    stage: int = 1,
    **kwargs,
) -> MultimodalLLM:
    """
    Factory function to create multimodal model.

    Args:
        model_size: LLM size ("125m", "350m", "1.3b", etc.)
        vision_encoder: Pretrained encoder name or None
        image_size: Input image resolution
        stage: Training stage (1=alignment, 2=instruction, 3=full)
    """
    MODEL_CONFIGS = {
        "125m": dict(n_layers=12, n_heads=12, d_model=768),
        "350m": dict(n_layers=24, n_heads=16, d_model=1024),
        "760m": dict(n_layers=24, n_heads=16, d_model=1536),
        "1.3b": dict(n_layers=24, n_heads=32, d_model=2048),
        "2.7b": dict(n_layers=32, n_heads=32, d_model=2560),
        "6.7b": dict(n_layers=32, n_heads=32, d_model=4096),
        "13b":  dict(n_layers=40, n_heads=40, d_model=5120),
    }

    llm_kwargs = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["350m"])
    llm_config = ModelConfig(**llm_kwargs, **kwargs)

    vision_config = VisionConfig(
        image_size=image_size,
        llm_hidden_size=llm_config.d_model,
    )

    mm_config = MultimodalConfig(
        llm_config=llm_config,
        vision_config=vision_config,
        pretrained_vision_encoder=vision_encoder,
    )

    model = MultimodalLLM(mm_config)
    model.set_stage(stage)

    params = model.num_parameters()
    print(f"\nModel parameters:")
    for k, v in params.items():
        print(f"  {k}: {v / 1e6:.1f}M")

    return model
