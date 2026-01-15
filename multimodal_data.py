"""
Multimodal Datasets for all training stages.

Supports:
  - Stage 1: Image-caption pairs for alignment pre-training
  - Stage 2: Multimodal instruction tuning (image/video + chat)
  - Stage 3: Multimodal preference data (DPO) and prompts (GRPO)
  - Multi-image and video conversations
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sft import ChatTokenizer


# =============================================================================
# Image transforms (minimal, no torchvision dependency)
# =============================================================================
def normalize_image(image: torch.Tensor, mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """Normalize image tensor. image: (C, H, W) in [0, 1]."""
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    return (image - mean) / std


def load_image(path: str, image_size: int = 448) -> torch.Tensor:
    """Load image from file and resize. Returns (C, H, W) in [0, 1]."""
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, (image_size, image_size))
        img = TF.to_tensor(img)
        return normalize_image(img)
    except ImportError:
        # Fallback: random tensor
        return torch.randn(3, image_size, image_size)


def load_video(path: str, max_frames: int = 32, image_size: int = 448) -> torch.Tensor:
    """Load video and sample frames. Returns (T, C, H, W)."""
    try:
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge("torch")
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        indices = torch.linspace(0, total - 1, min(max_frames, total)).long().tolist()
        frames = vr.get_batch(indices).permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(frames, size=(image_size, image_size), mode="bicubic", align_corners=False)
        for i in range(frames.shape[0]):
            frames[i] = normalize_image(frames[i])
        return frames
    except ImportError:
        return torch.randn(min(max_frames, 8), 3, image_size, image_size)


# =============================================================================
# Stage 1: Image-Caption Alignment Dataset
# =============================================================================
class AlignmentDataset(Dataset):
    """
    Image-caption pairs for vision-language alignment pre-training.

    Data format (JSONL):
        {"image": "path/to/image.jpg", "caption": "A cat sitting on a mat"}
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, data_path: str, image_dir: str, tokenizer: ChatTokenizer,
                 image_size: int = 448, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.image_dir = image_dir
        self.examples = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.examples.append(data)

        print(f"Loaded {len(self.examples)} alignment examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        caption = example.get("caption", example.get("text", ""))
        image_path = example.get("image", example.get("image_path", ""))

        # Load image
        full_path = os.path.join(self.image_dir, image_path) if self.image_dir else image_path
        if os.path.exists(full_path):
            pixel_values = load_image(full_path, self.image_size)
        else:
            pixel_values = torch.randn(3, self.image_size, self.image_size)

        # Tokenize: "<image> caption"
        prompt = f"{self.IMAGE_TOKEN} {caption}"
        tokens = self.tokenizer.encode(prompt)[:self.max_seq_len]
        input_ids = torch.tensor([self.tokenizer.bos_id] + tokens, dtype=torch.long)

        # Labels: train on caption tokens, mask image placeholder
        image_token_pos = 1  # After BOS
        labels = input_ids.clone()
        # Mask everything up to and including the image placeholder region
        labels[:image_token_pos + 1] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_token_pos": torch.tensor(image_token_pos, dtype=torch.long),
        }


# =============================================================================
# Stage 2: Multimodal Instruction Tuning Dataset
# =============================================================================
class MultimodalSFTDataset(Dataset):
    """
    Multimodal instruction tuning with image/video + chat.

    Data format (JSONL):
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
    """

    def __init__(self, data_path: str, media_dir: str, tokenizer: ChatTokenizer,
                 image_size: int = 448, max_seq_len: int = 2048, max_frames: int = 32):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.max_frames = max_frames
        self.media_dir = media_dir
        self.examples = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} multimodal SFT examples")

    def __len__(self):
        return len(self.examples)

    def _load_media(self, example):
        """Load image(s) or video from example."""
        pixel_values = None
        video_frames = None
        media_type = "none"

        if "image" in example:
            path = os.path.join(self.media_dir, example["image"]) if self.media_dir else example["image"]
            if os.path.exists(path):
                pixel_values = load_image(path, self.image_size).unsqueeze(0)  # (1, C, H, W)
            else:
                pixel_values = torch.randn(1, 3, self.image_size, self.image_size)
            media_type = "image"

        elif "images" in example:
            imgs = []
            for img_path in example["images"]:
                path = os.path.join(self.media_dir, img_path) if self.media_dir else img_path
                if os.path.exists(path):
                    imgs.append(load_image(path, self.image_size))
                else:
                    imgs.append(torch.randn(3, self.image_size, self.image_size))
            pixel_values = torch.stack(imgs)  # (N, C, H, W)
            media_type = "multi_image"

        elif "video" in example:
            path = os.path.join(self.media_dir, example["video"]) if self.media_dir else example["video"]
            if os.path.exists(path):
                video_frames = load_video(path, self.max_frames, self.image_size)
            else:
                video_frames = torch.randn(8, 3, self.image_size, self.image_size)
            media_type = "video"

        return pixel_values, video_frames, media_type

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example.get("messages", [])

        # Load media
        pixel_values, video_frames, media_type = self._load_media(example)

        # Tokenize conversation with loss masking
        encoded = self.tokenizer.encode_chat(messages)
        input_ids = encoded["input_ids"][:self.max_seq_len]
        loss_mask = encoded["loss_mask"][:self.max_seq_len]

        # Labels: only train on assistant tokens
        labels = input_ids[1:] + [self.tokenizer.pad_id]
        loss_mask_shifted = loss_mask[1:] + [0]
        labels = [l if m == 1 else -100 for l, m in zip(labels, loss_mask_shifted)]

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "media_type": media_type,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if video_frames is not None:
            result["video_frames"] = video_frames

        return result


# =============================================================================
# Stage 3a: Multimodal DPO Dataset
# =============================================================================
class MultimodalDPODataset(Dataset):
    """
    Preference pairs with multimodal context.

    Data format (JSONL):
        {"image": "path.jpg",
         "chosen": [{"role": "user", "content": "<image>\nDescribe"}, {"role": "assistant", "content": "Good answer"}],
         "rejected": [{"role": "user", "content": "<image>\nDescribe"}, {"role": "assistant", "content": "Bad answer"}]}
    """

    def __init__(self, data_path: str, media_dir: str, tokenizer: ChatTokenizer,
                 image_size: int = 448, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.media_dir = media_dir
        self.pairs = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("chosen") and data.get("rejected"):
                    self.pairs.append(data)

        print(f"Loaded {len(self.pairs)} multimodal DPO pairs")

    def __len__(self):
        return len(self.pairs)

    def _encode(self, messages):
        encoded = self.tokenizer.encode_chat(messages)
        ids = encoded["input_ids"][:self.max_seq_len]
        mask = encoded["loss_mask"][:self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)

    def _load_image(self, example):
        if "image" in example:
            path = os.path.join(self.media_dir, example["image"]) if self.media_dir else example["image"]
            if os.path.exists(path):
                return load_image(path, self.image_size)
        return torch.randn(3, self.image_size, self.image_size)

    def __getitem__(self, idx):
        example = self.pairs[idx]
        pixel_values = self._load_image(example)

        chosen_ids, chosen_mask = self._encode(example["chosen"])
        rejected_ids, rejected_mask = self._encode(example["rejected"])

        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "pixel_values": pixel_values,
        }


# =============================================================================
# Stage 3b: Multimodal GRPO Dataset
# =============================================================================
class MultimodalGRPODataset(Dataset):
    """
    Prompts with optional images for multimodal GRPO.

    Data format (JSONL):
        {"image": "path.jpg", "prompt": "What is shown in this image?", "answer": "A cat"}
        {"image": "chart.png", "prompt": "What is the highest value?", "answer": "42"}
    """

    def __init__(self, data_path: str, media_dir: str, tokenizer: ChatTokenizer,
                 image_size: int = 448, max_prompt_len: int = 512):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_prompt_len = max_prompt_len
        self.media_dir = media_dir
        self.items = []

        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

        print(f"Loaded {len(self.items)} multimodal GRPO prompts")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Prompt
        prompt = item.get("prompt", "")
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        encoded = self.tokenizer.encode_chat(prompt)
        ids = encoded["input_ids"][:self.max_prompt_len]

        # Image
        pixel_values = None
        if "image" in item:
            path = os.path.join(self.media_dir, item["image"]) if self.media_dir else item["image"]
            if os.path.exists(path):
                pixel_values = load_image(path, self.image_size)
            else:
                pixel_values = torch.randn(3, self.image_size, self.image_size)

        result = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "answer": item.get("answer"),
            "prompt_text": item.get("prompt", ""),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result


# =============================================================================
# Collators
# =============================================================================
class AlignmentCollator:
    """Collator for alignment pre-training."""

    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        max_seq = min(max(len(b["input_ids"]) for b in batch), self.max_len)

        input_ids = []
        labels = []
        pixel_values = []
        image_positions = []

        for b in batch:
            ids = b["input_ids"][:max_seq]
            lab = b["labels"][:max_seq]
            pad_len = max_seq - len(ids)

            input_ids.append(F.pad(ids, (0, pad_len), value=self.pad_id))
            labels.append(F.pad(lab, (0, pad_len), value=-100))
            pixel_values.append(b["pixel_values"])
            image_positions.append(b["image_token_pos"])

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack(pixel_values),
            "image_positions": torch.stack(image_positions),
        }


class MultimodalSFTCollator:
    """Collator for multimodal SFT with mixed media types."""

    def __init__(self, pad_id: int, max_len: int, image_size: int = 448):
        self.pad_id = pad_id
        self.max_len = max_len
        self.image_size = image_size

    def __call__(self, batch):
        max_seq = min(max(len(b["input_ids"]) for b in batch), self.max_len)

        input_ids = []
        labels = []
        pixel_values_list = []
        video_frames_list = []
        has_image = False
        has_video = False

        for b in batch:
            ids = b["input_ids"][:max_seq]
            lab = b["labels"][:max_seq]
            pad_len = max_seq - len(ids)

            input_ids.append(F.pad(ids, (0, pad_len), value=self.pad_id))
            labels.append(F.pad(lab, (0, pad_len), value=-100))

            if "pixel_values" in b:
                pixel_values_list.append(b["pixel_values"])
                has_image = True
            else:
                pixel_values_list.append(torch.zeros(1, 3, self.image_size, self.image_size))

            if "video_frames" in b:
                video_frames_list.append(b["video_frames"])
                has_video = True

        result = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
        }

        if has_image:
            # Stack images (use first image per sample for simplicity)
            imgs = []
            for pv in pixel_values_list:
                if pv.dim() == 4:
                    imgs.append(pv[0])  # Take first image
                else:
                    imgs.append(pv)
            result["pixel_values"] = torch.stack(imgs)

        if has_video:
            # Pad video frames to same length
            max_frames = max(vf.shape[0] for vf in video_frames_list)
            padded_videos = []
            for vf in video_frames_list:
                if vf.shape[0] < max_frames:
                    pad = torch.zeros(
                        max_frames - vf.shape[0], 3, self.image_size, self.image_size,
                    )
                    vf = torch.cat([vf, pad], dim=0)
                padded_videos.append(vf)
            result["video_frames"] = torch.stack(padded_videos)

        return result


# =============================================================================
# Synthetic Data Generator (for testing)
# =============================================================================
def generate_synthetic_multimodal_data(
    output_dir: str = "data",
    num_alignment: int = 1000,
    num_sft: int = 500,
    num_dpo: int = 200,
    num_grpo: int = 200,
):
    """Generate synthetic data files for testing the multimodal pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    captions = [
        "A beautiful sunset over the ocean with orange and purple clouds.",
        "A cat sitting on a windowsill looking outside.",
        "A modern city skyline at night with illuminated buildings.",
        "A group of people hiking through a mountain trail.",
        "A close-up of a flower with dewdrops on its petals.",
        "A dog playing fetch in a green park.",
        "A bowl of fresh fruit on a wooden table.",
        "An astronaut floating in space with Earth in the background.",
    ]

    # Alignment data
    with open(os.path.join(output_dir, "mm_alignment.jsonl"), "w") as f:
        for i in range(num_alignment):
            f.write(json.dumps({
                "image": f"images/img_{i:06d}.jpg",
                "caption": random.choice(captions),
            }) + "\n")

    # SFT data
    questions = [
        "Describe what you see in this image in detail.",
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "Describe the mood or atmosphere of this image.",
        "What objects can you identify in this image?",
    ]

    with open(os.path.join(output_dir, "mm_sft.jsonl"), "w") as f:
        for i in range(num_sft):
            question = random.choice(questions)
            answer = random.choice(captions) + " The image conveys a sense of " + \
                     random.choice(["tranquility", "excitement", "wonder", "joy"]) + "."
            f.write(json.dumps({
                "image": f"images/img_{i:06d}.jpg",
                "messages": [
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": answer},
                ],
            }) + "\n")

        # Video examples
        for i in range(num_sft // 10):
            f.write(json.dumps({
                "video": f"videos/vid_{i:04d}.mp4",
                "messages": [
                    {"role": "user", "content": "<video>\nDescribe what happens in this video."},
                    {"role": "assistant", "content": "The video shows a sequence of events..."},
                ],
            }) + "\n")

    # DPO data
    with open(os.path.join(output_dir, "mm_dpo.jsonl"), "w") as f:
        for i in range(num_dpo):
            question = random.choice(questions)
            f.write(json.dumps({
                "image": f"images/img_{i:06d}.jpg",
                "chosen": [
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": random.choice(captions) + " It's a detailed description."},
                ],
                "rejected": [
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": "I see an image. It has stuff in it."},
                ],
            }) + "\n")

    # GRPO data
    with open(os.path.join(output_dir, "mm_grpo.jsonl"), "w") as f:
        for i in range(num_grpo):
            f.write(json.dumps({
                "image": f"charts/chart_{i:04d}.png",
                "prompt": f"<image>\nWhat is the value at position {random.randint(1, 10)}?",
                "answer": str(random.randint(10, 100)),
            }) + "\n")

    print(f"Generated synthetic multimodal data in {output_dir}/")
    print(f"  mm_alignment.jsonl: {num_alignment} examples")
    print(f"  mm_sft.jsonl:       {num_sft} examples")
    print(f"  mm_dpo.jsonl:       {num_dpo} pairs")
    print(f"  mm_grpo.jsonl:      {num_grpo} prompts")


if __name__ == "__main__":
    generate_synthetic_multimodal_data()
