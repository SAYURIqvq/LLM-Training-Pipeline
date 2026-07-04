import json

import pytest

from generate_sample_data import (
    generate_preference_data,
    generate_prompt_data,
    generate_sft_data,
)
from pipeline_validation import (
    ValidationError,
    validate_deepspeed_config,
    validate_preference_rows,
    validate_prompt_rows,
    validate_sft_rows,
)


def test_generated_sample_data_matches_expected_schemas():
    assert validate_sft_rows(generate_sft_data(3)) == 3
    assert validate_preference_rows(generate_preference_data(2)) == 2
    assert validate_prompt_rows(generate_prompt_data(4)) == 4


def test_sft_validation_requires_assistant_final_message():
    rows = [{"messages": [{"role": "user", "content": "hello"}]}]

    with pytest.raises(ValidationError):
        validate_sft_rows(rows)


def test_preference_validation_requires_both_pairs():
    rows = [{"chosen": [{"role": "assistant", "content": "ok"}]}]

    with pytest.raises(ValidationError):
        validate_preference_rows(rows)


def test_deepspeed_config_validation(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({
        "train_batch_size": 8,
        "zero_optimization": {"stage": 2},
        "optimizer": {"type": "AdamW"},
    }))

    config = validate_deepspeed_config(path)

    assert config["zero_optimization"]["stage"] == 2


def test_deepspeed_config_rejects_invalid_stage(tmp_path):
    path = tmp_path / "bad_ds.json"
    path.write_text(json.dumps({
        "train_batch_size": 8,
        "zero_optimization": {"stage": 9},
        "optimizer": {"type": "AdamW"},
    }))

    with pytest.raises(ValidationError):
        validate_deepspeed_config(path)
