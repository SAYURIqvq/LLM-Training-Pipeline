"""CPU training/checkpoint smoke test using the actual GPTModel (no downloads)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from train import GPTModel, ModelConfig


def train_step(model, optimizer, tokens):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(tokens, labels=tokens)
    if not torch.isfinite(loss):
        raise AssertionError("loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("missing or non-finite gradients")
    if not any(torch.count_nonzero(g) for g in gradients):
        raise AssertionError("all gradients are zero")
    optimizer.step()
    return loss.item()


def run_smoke(output_dir, steps=3):
    if steps < 1:
        raise ValueError("steps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2,
                         n_heads=4, n_kv_heads=2, d_model=32, d_ff=64,
                         use_flash_attn=False, use_te=False, dropout=0.1)
    model = GPTModel(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    initial = model.tok_emb.weight.detach().clone()
    losses = [train_step(model, optimizer, tokens) for _ in range(steps)]
    if torch.equal(initial, model.tok_emb.weight):
        raise AssertionError("parameters did not update")
    checkpoint = output_dir / 'smoke_checkpoint.pt'
    torch.save({'config': asdict(config), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                'step': steps}, checkpoint)
    uninterrupted_loss = train_step(model, optimizer, tokens)

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    resumed = GPTModel(ModelConfig(**saved['config'])).train()
    resumed.load_state_dict(saved['model'], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    resumed_loss = train_step(resumed, resumed_optimizer, tokens)
    if uninterrupted_loss != resumed_loss:
        raise AssertionError("resumed loss differs from uninterrupted training")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    result = {'device': 'cpu', 'torch_version': str(torch.__version__),
              'steps_before_checkpoint': steps, 'losses': losses,
              'resumed_loss': resumed_loss, 'parameters_updated': True,
              'resume_matches_uninterrupted': True,
              'scope': 'GPTModel + SDPA + AdamW; not DeepSpeed/FP8/FlashAttention validation'}
    (output_dir / 'smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/smoke')
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir, args.steps), indent=2))
