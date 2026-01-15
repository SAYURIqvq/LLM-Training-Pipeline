"""
Vision Encoder + Projector for Multimodal LLM.

Supports:
  - SigLIP / CLIP / ViT backbone (via timm or custom)
  - Dynamic resolution with tile-based processing
  - Pixel shuffle token compression (4x or 9x reduction)
  - MLP projector to align vision features with LLM embedding space

Architecture: Image → ViT Encoder → Pixel Shuffle → MLP Projector → Visual Tokens
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Vision Config
# =============================================================================
@dataclass
class VisionConfig:
    # Encoder
    image_size: int = 448           # Input resolution per tile
    patch_size: int = 14            # ViT patch size
    in_channels: int = 3
    hidden_size: int = 1152         # ViT hidden dim (SigLIP-400M ~ 1152)
    num_layers: int = 27            # ViT depth
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Projector
    projector_type: str = "mlp"     # "mlp" or "cross_attn"
    projector_depth: int = 2        # Number of MLP layers
    llm_hidden_size: int = 1024     # Must match LLM d_model

    # Token compression
    pixel_shuffle_ratio: int = 2    # 2 = 4x reduction, 3 = 9x reduction
    use_pixel_shuffle: bool = True

    # Dynamic resolution
    min_tiles: int = 1
    max_tiles: int = 12             # Max number of tiles per image
    tile_size: int = 448

    # Video
    max_frames: int = 64            # Max frames for video
    fps_sample: float = 1.0         # Frames per second to sample

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_visual_tokens_per_tile(self):
        """Tokens per tile after pixel shuffle."""
        n = self.num_patches
        if self.use_pixel_shuffle:
            n = n // (self.pixel_shuffle_ratio ** 2)
        return n


# =============================================================================
# Vision Transformer (ViT) Encoder
# =============================================================================
class VisionPatchEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        num_patches = config.num_patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, config.hidden_size) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, N, D)"""
        x = self.proj(x)                    # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        x = x + self.position_embedding[:, :x.shape[1]]
        return x


class VisionTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Simple ViT encoder. Can be replaced with pretrained SigLIP/CLIP."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbedding(config)
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                config.hidden_size, config.num_heads,
                config.mlp_ratio, config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) or (B*T, C, H, W) for tiles/frames
        Returns:
            (B, N, D) patch features
        """
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# =============================================================================
# Pretrained Encoder Wrapper (SigLIP / CLIP via timm)
# =============================================================================
class PretrainedVisionEncoder(nn.Module):
    """Load pretrained vision encoders from timm or HuggingFace."""

    def __init__(self, model_name: str = "vit_so400m_patch14_siglip_384",
                 hidden_size: int = 1152, freeze: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.freeze = freeze

        try:
            import timm
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
            )
            # Get feature dim
            if hasattr(self.model, 'embed_dim'):
                self.actual_hidden_size = self.model.embed_dim
            else:
                self.actual_hidden_size = hidden_size

            if self.freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()

            print(f"Loaded pretrained vision encoder: {model_name} "
                  f"(hidden={self.actual_hidden_size}, frozen={freeze})")

        except (ImportError, Exception) as e:
            print(f"Could not load pretrained encoder ({e}), using random init ViT")
            self.model = None
            self.actual_hidden_size = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            if self.freeze:
                with torch.no_grad():
                    features = self.model.forward_features(pixel_values)
            else:
                features = self.model.forward_features(pixel_values)
            return features
        return pixel_values


# =============================================================================
# Pixel Shuffle Token Compression
# =============================================================================
class PixelShuffle2D(nn.Module):
    """
    Reduce visual tokens by pixel shuffle (spatial merging).
    Ratio=2: 4x token reduction (e.g., 1024 → 256)
    Ratio=3: 9x token reduction (e.g., 729 → 81)
    """

    def __init__(self, hidden_size: int, ratio: int = 2):
        super().__init__()
        self.ratio = ratio
        # After merging ratio^2 tokens, project back to hidden_size
        self.proj = nn.Linear(hidden_size * ratio * ratio, hidden_size)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        x: (B, H*W, D)
        h, w: spatial dimensions of the token grid
        Returns: (B, H//r * W//r, D)
        """
        B, N, D = x.shape
        r = self.ratio

        # Ensure dimensions are divisible
        new_h = (h // r) * r
        new_w = (w // r) * r
        if new_h * new_w < N:
            x = x[:, :new_h * new_w]

        x = x.view(B, new_h // r, r, new_w // r, r, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H', W', r, r, D)
        x = x.view(B, (new_h // r) * (new_w // r), r * r * D)  # (B, N', r*r*D)
        x = self.proj(x)  # (B, N', D)
        return x


# =============================================================================
# Vision-Language Projector
# =============================================================================
class MLPProjector(nn.Module):
    """MLP projector to align vision features with LLM embedding space."""

    def __init__(self, vision_hidden: int, llm_hidden: int, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = vision_hidden
        for i in range(depth):
            out_dim = llm_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionProjector(nn.Module):
    """Cross-attention based projector with learnable queries."""

    def __init__(self, vision_hidden: int, llm_hidden: int,
                 num_queries: int = 64, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_hidden) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    llm_hidden, num_heads, kdim=vision_hidden, vdim=vision_hidden,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(llm_hidden),
                "ffn": nn.Sequential(
                    nn.Linear(llm_hidden, llm_hidden * 4),
                    nn.GELU(),
                    nn.Linear(llm_hidden * 4, llm_hidden),
                ),
                "norm2": nn.LayerNorm(llm_hidden),
            }))

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        B = vision_features.shape[0]
        queries = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            h = layer["norm1"](queries)
            h, _ = layer["cross_attn"](h, vision_features, vision_features)
            queries = queries + h
            queries = queries + layer["ffn"](layer["norm2"](queries))

        return queries


# =============================================================================
# Dynamic Resolution Tiling
# =============================================================================
class DynamicTiler:
    """Split high-resolution images into tiles for processing."""

    def __init__(self, tile_size: int = 448, min_tiles: int = 1, max_tiles: int = 12):
        self.tile_size = tile_size
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles

    def find_best_tiling(self, width: int, height: int) -> Tuple[int, int]:
        """Find optimal grid (rows, cols) for the given image dimensions."""
        aspect = width / height
        best_tiles = (1, 1)
        best_waste = float("inf")

        for total in range(self.min_tiles, self.max_tiles + 1):
            for rows in range(1, total + 1):
                cols = total // rows
                if rows * cols > self.max_tiles or cols == 0:
                    continue
                tile_aspect = cols / rows
                waste = abs(tile_aspect - aspect)
                if waste < best_waste:
                    best_waste = waste
                    best_tiles = (rows, cols)

        return best_tiles

    def tile_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Split image into tiles.
        image: (C, H, W)
        Returns: (num_tiles, C, tile_size, tile_size), (rows, cols)
        """
        C, H, W = image.shape
        rows, cols = self.find_best_tiling(W, H)

        # Resize to fit the grid
        target_h = rows * self.tile_size
        target_w = cols * self.tile_size
        image = F.interpolate(
            image.unsqueeze(0), size=(target_h, target_w),
            mode="bicubic", align_corners=False,
        ).squeeze(0)

        # Split into tiles
        tiles = []
        for r in range(rows):
            for c in range(cols):
                y0 = r * self.tile_size
                x0 = c * self.tile_size
                tile = image[:, y0:y0 + self.tile_size, x0:x0 + self.tile_size]
                tiles.append(tile)

        # Add a thumbnail (global view)
        thumbnail = F.interpolate(
            image.unsqueeze(0), size=(self.tile_size, self.tile_size),
            mode="bicubic", align_corners=False,
        ).squeeze(0)
        tiles.append(thumbnail)

        return torch.stack(tiles), (rows, cols)


# =============================================================================
# Video Frame Sampler
# =============================================================================
class VideoProcessor:
    """Extract and process frames from video tensors."""

    def __init__(self, max_frames: int = 64, fps_sample: float = 1.0,
                 image_size: int = 448):
        self.max_frames = max_frames
        self.fps_sample = fps_sample
        self.image_size = image_size

    def sample_frames(self, video: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
        """
        Uniformly sample frames from video tensor.
        video: (T, C, H, W)
        Returns: (num_frames, C, H, W)
        """
        T = video.shape[0]
        total_seconds = T / fps
        target_frames = min(int(total_seconds * self.fps_sample), self.max_frames)
        target_frames = max(target_frames, 1)

        if target_frames >= T:
            indices = list(range(T))
        else:
            indices = torch.linspace(0, T - 1, target_frames).long().tolist()

        frames = video[indices]

        # Resize
        if frames.shape[-2] != self.image_size or frames.shape[-1] != self.image_size:
            frames = F.interpolate(
                frames, size=(self.image_size, self.image_size),
                mode="bicubic", align_corners=False,
            )

        return frames

    def process_video_file(self, video_path: str) -> torch.Tensor:
        """Load and sample frames from a video file."""
        try:
            import decord
            from decord import VideoReader, cpu
            decord.bridge.set_bridge("torch")

            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            target = min(int(total_frames / fps * self.fps_sample), self.max_frames)
            target = max(target, 1)

            indices = torch.linspace(0, total_frames - 1, target).long().tolist()
            frames = vr.get_batch(indices)  # (T, H, W, C)
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

            if frames.shape[-2] != self.image_size or frames.shape[-1] != self.image_size:
                frames = F.interpolate(
                    frames, size=(self.image_size, self.image_size),
                    mode="bicubic", align_corners=False,
                )
            return frames

        except ImportError:
            print("decord not installed. Install with: pip install decord")
            # Return dummy frames
            return torch.randn(4, 3, self.image_size, self.image_size)


# =============================================================================
# Complete Vision Module
# =============================================================================
class VisionModule(nn.Module):
    """
    Complete vision processing pipeline:
    Image/Video → ViT Encoder → Pixel Shuffle → MLP Projector → Visual Tokens
    """

    def __init__(self, config: VisionConfig, pretrained_encoder: str = None):
        super().__init__()
        self.config = config

        # Vision encoder
        if pretrained_encoder:
            self.encoder = PretrainedVisionEncoder(
                pretrained_encoder, config.hidden_size, freeze=True,
            )
            encoder_hidden = self.encoder.actual_hidden_size
        else:
            self.encoder = VisionTransformer(config)
            encoder_hidden = config.hidden_size

        # Pixel shuffle compression
        self.pixel_shuffle = None
        if config.use_pixel_shuffle:
            self.pixel_shuffle = PixelShuffle2D(encoder_hidden, config.pixel_shuffle_ratio)

        # Projector
        proj_input_dim = encoder_hidden
        if config.projector_type == "mlp":
            self.projector = MLPProjector(
                proj_input_dim, config.llm_hidden_size, config.projector_depth,
            )
        elif config.projector_type == "cross_attn":
            self.projector = CrossAttentionProjector(
                proj_input_dim, config.llm_hidden_size,
            )
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        # Dynamic tiling
        self.tiler = DynamicTiler(config.tile_size, config.min_tiles, config.max_tiles)

        # Video processor
        self.video_processor = VideoProcessor(
            config.max_frames, config.fps_sample, config.image_size,
        )

        # Special tokens (learnable)
        self.image_start_token = nn.Parameter(torch.randn(1, 1, config.llm_hidden_size) * 0.02)
        self.image_end_token = nn.Parameter(torch.randn(1, 1, config.llm_hidden_size) * 0.02)
        self.frame_separator = nn.Parameter(torch.randn(1, 1, config.llm_hidden_size) * 0.02)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image or batch of tiles.
        pixel_values: (B, C, H, W) or (num_tiles, C, H, W)
        Returns: (B, num_visual_tokens, llm_hidden)
        """
        # ViT forward
        features = self.encoder(pixel_values)  # (B, N, D)

        # Pixel shuffle
        if self.pixel_shuffle is not None:
            h = w = int(math.sqrt(features.shape[1]))
            features = self.pixel_shuffle(features, h, w)

        # Project to LLM space
        visual_tokens = self.projector(features)  # (B, N', llm_hidden)
        return visual_tokens

    def encode_image_dynamic(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image with dynamic resolution tiling.
        image: (C, H, W) single image
        Returns: (1, total_visual_tokens, llm_hidden) with start/end tokens
        """
        tiles, (rows, cols) = self.tiler.tile_image(image)  # (num_tiles+1, C, H, W)

        # Encode all tiles at once
        tile_features = self.encode_image(tiles)  # (num_tiles+1, N, D)

        # Flatten all tile tokens into a sequence
        all_tokens = tile_features.view(1, -1, tile_features.shape[-1])

        # Add start/end tokens
        start = self.image_start_token.expand(1, -1, -1)
        end = self.image_end_token.expand(1, -1, -1)
        all_tokens = torch.cat([start, all_tokens, end], dim=1)

        return all_tokens

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames.
        frames: (T, C, H, W)
        Returns: (1, total_tokens, llm_hidden) with frame separators
        """
        T = frames.shape[0]

        # Encode all frames as a batch
        frame_features = self.encode_image(frames)  # (T, N, D)

        # Interleave frame features with separator tokens
        tokens_list = [self.image_start_token.expand(1, -1, -1)]
        for i in range(T):
            tokens_list.append(frame_features[i:i + 1])  # (1, N, D)
            if i < T - 1:
                tokens_list.append(self.frame_separator.expand(1, -1, -1))
        tokens_list.append(self.image_end_token.expand(1, -1, -1))

        all_tokens = torch.cat(tokens_list, dim=1)  # (1, total, D)
        return all_tokens

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Process image or video input.

        Args:
            pixel_values: (B, C, H, W) for simple images, or pre-tiled
            video_frames: (B, T, C, H, W) video frames

        Returns:
            visual_tokens: (B, num_tokens, llm_hidden)
        """
        if pixel_values is not None:
            if pixel_values.dim() == 4:
                return self.encode_image(pixel_values)
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        if video_frames is not None:
            B = video_frames.shape[0]
            all_tokens = []
            for b in range(B):
                tokens = self.encode_video(video_frames[b])  # (1, N, D)
                all_tokens.append(tokens)
            # Pad to max length
            max_len = max(t.shape[1] for t in all_tokens)
            padded = []
            for t in all_tokens:
                if t.shape[1] < max_len:
                    pad = torch.zeros(
                        1, max_len - t.shape[1], t.shape[2],
                        device=t.device, dtype=t.dtype,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.cat(padded, dim=0)

        return None
