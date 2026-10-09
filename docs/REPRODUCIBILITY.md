# Reproducibility and validation scope

## CPU path

Python 3.11–3.12 and the versions in `requirements-smoke.txt` run the repository's actual `GPTModel` with PyTorch SDPA, GQA, RoPE, RMSNorm, SwiGLU and AdamW. No model or dataset download is required. `smoke_train.py` checks finite loss/gradients, parameter updates, and exact next-step equivalence after restoring model, optimizer and RNG state. The causal attention regression test verifies that future tokens cannot change earlier logits.

```bash
python -m pip install -r requirements-smoke.txt
python -m pytest tests/test_pipeline_validation.py tests/test_training_smoke.py -q
python smoke_train.py --output-dir outputs/smoke --steps 3
```

The emitted JSON records the actual PyTorch version and losses. This is a correctness smoke test, not a convergence or performance benchmark.

## GPU paths (separate validation required)

The original entry points require a CUDA-compatible PyTorch build and DeepSpeed; scripts using Transformer Engine require its matching CUDA libraries. FlashAttention 2 is required when that backend is selected; FlashAttention 3 additionally uses `kernels` and compatible hardware. Dataset/tokenizer paths use `datasets` and `tiktoken`; some paths also require `transformers`, Pillow or other optional libraries.

There is no verified GPU lockfile in this change. Do not treat the CPU requirements as sufficient for SFT/PPO/GRPO or multimodal scripts. Install GPU packages according to the selected CUDA/PyTorch combination, then record the resolved environment before reporting results. Automatic fallback does not silently disable requested GPU features: missing requested backends raise import errors.

## Evidence required for performance claims

Record commit SHA, package freeze, GPU model/count, driver/CUDA versions, model dimensions, dtype, attention backend, sequence length, micro/global batch, accumulation, optimizer and ZeRO stage. Run warm-up and repeated measurements with the same token budget. Publish tokens/sec, peak memory, loss and raw logs for both baseline and optimized runs. FP8 acceleration is currently a capability to validate, not a demonstrated 2× speedup in this repository.

## Limits

CPU CI does not establish GPU kernel correctness, distributed checkpoint equivalence, training convergence, SFT masking correctness or alignment quality. The CPU checkpoint format is a small PyTorch test artifact, not a conversion tool for DeepSpeed checkpoints.
