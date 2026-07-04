"""Lightweight validation utilities for training data and DeepSpeed configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VALID_ROLES = {"system", "user", "assistant"}


class ValidationError(ValueError):
    """Raised when a training artifact is structurally invalid."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValidationError(f"{path}:{line_no}: expected JSON object")
            rows.append(item)
    return rows


def validate_messages(messages: Any, *, path: str = "messages") -> None:
    if not isinstance(messages, list) or not messages:
        raise ValidationError(f"{path} must be a non-empty list")

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"{path}[{idx}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            raise ValidationError(f"{path}[{idx}].role must be one of {sorted(VALID_ROLES)}")
        if not isinstance(content, str) or not content.strip():
            raise ValidationError(f"{path}[{idx}].content must be a non-empty string")


def validate_sft_rows(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for idx, row in enumerate(rows):
        validate_messages(row.get("messages"), path=f"row[{idx}].messages")
        if row["messages"][-1]["role"] != "assistant":
            raise ValidationError(f"row[{idx}].messages must end with an assistant response")
        count += 1
    return count


def validate_preference_rows(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for idx, row in enumerate(rows):
        validate_messages(row.get("chosen"), path=f"row[{idx}].chosen")
        validate_messages(row.get("rejected"), path=f"row[{idx}].rejected")
        if row["chosen"][-1]["role"] != "assistant" or row["rejected"][-1]["role"] != "assistant":
            raise ValidationError(f"row[{idx}] chosen/rejected must end with assistant responses")
        count += 1
    return count


def validate_prompt_rows(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for idx, row in enumerate(rows):
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValidationError(f"row[{idx}].prompt must be non-empty")
        else:
            validate_messages(prompt, path=f"row[{idx}].prompt")
        count += 1
    return count


def validate_deepspeed_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    required = ["train_batch_size", "zero_optimization", "optimizer"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValidationError(f"{path}: missing required keys: {missing}")

    stage = config["zero_optimization"].get("stage")
    if stage not in {0, 1, 2, 3}:
        raise ValidationError(f"{path}: zero_optimization.stage must be 0, 1, 2, or 3")

    if config["train_batch_size"] <= 0:
        raise ValidationError(f"{path}: train_batch_size must be positive")

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training data and DeepSpeed configs.")
    parser.add_argument("--sft")
    parser.add_argument("--preference")
    parser.add_argument("--prompts")
    parser.add_argument("--deepspeed", action="append", default=[])
    args = parser.parse_args()

    if args.sft:
        print(f"SFT rows: {validate_sft_rows(load_jsonl(args.sft))}")
    if args.preference:
        print(f"Preference rows: {validate_preference_rows(load_jsonl(args.preference))}")
    if args.prompts:
        print(f"Prompt rows: {validate_prompt_rows(load_jsonl(args.prompts))}")
    for config_path in args.deepspeed:
        config = validate_deepspeed_config(config_path)
        print(f"DeepSpeed config OK: {config_path} (ZeRO-{config['zero_optimization']['stage']})")


if __name__ == "__main__":
    main()
