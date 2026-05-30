# Attention Is All You Need — from scratch

A PyTorch implementation of the original Transformer (Vaswani et al., 2017),
built one component at a time without relying on `nn.Transformer`,
`nn.MultiheadAttention`, or `nn.LayerNorm`. Plain `nn.Linear`, `nn.Embedding`,
and tensor ops only.

Target hardware: MacBook M4, 16 GB unified memory. Runs on the MPS backend.

## Status

Boilerplate scaffold. Every transformer module is stubbed with type-hinted
signatures, shape contracts in docstrings, and `NotImplementedError` bodies.
Tests pin the expected shapes; once a module is implemented its tests should
pass.

## Layout

```
src/transformer/   # one concept per file: attention, positional, layers, ...
tests/             # shape and behavior tests, one file per module
scripts/           # sanity_check.py, train_copy_task.py
```

## Setup

```bash
uv sync                                        # install torch + dev deps
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

## Run

```bash
uv run pytest                                  # collect & run tests
uv run python scripts/sanity_check.py          # forward + backward smoke test
uv run python scripts/train_copy_task.py       # toy copy-task training
uv run ruff check .                            # lint
```

## Reading order (suggested)

When filling in the stubs, follow the dependency order so each module's
tests can actually run:

1. `masking.py`  → `tests/test_masking.py`
2. `layernorm.py` → `tests/test_layernorm.py`
3. `positional.py` → `tests/test_positional.py`
4. `attention.py` (SDPA, then MultiHeadAttention) → `tests/test_attention.py`
5. `feedforward.py`
6. `layers.py` (SublayerConnection, EncoderLayer, DecoderLayer)
7. `encoder.py`, `decoder.py`
8. `transformer.py` → `tests/test_transformer_shapes.py`
9. `label_smoothing.py`, `optim.py`
10. `scripts/sanity_check.py` → confirm full forward + backward
11. `scripts/train_copy_task.py` → confirm the model can actually learn

Each numbered step is a self-contained chunk of work.
