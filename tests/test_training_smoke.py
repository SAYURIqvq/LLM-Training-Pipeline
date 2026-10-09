import torch
from smoke_train import run_smoke
from train import GPTModel, ModelConfig


def test_real_model_updates_and_resumes_exactly(tmp_path):
    result = run_smoke(tmp_path, steps=2)
    assert result['parameters_updated']
    assert result['resume_matches_uninterrupted']
    assert (tmp_path / 'smoke_result.json').exists()


def test_attention_is_causal():
    torch.manual_seed(3)
    model = GPTModel(ModelConfig(vocab_size=32, max_seq_len=8, n_layers=1,
                                n_heads=4, n_kv_heads=2, d_model=16,
                                use_flash_attn=False, use_te=False)).eval()
    tokens = torch.randint(0, 32, (1, 8))
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    with torch.no_grad():
        original, _ = model(tokens)
        altered, _ = model(changed)
    torch.testing.assert_close(original[:, :4], altered[:, :4], rtol=0, atol=0)
