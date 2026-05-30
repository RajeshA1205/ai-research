"""Toy sequence-copy task — the canonical 'is the Transformer wired up right?' check.

Run with:  uv run python scripts/train_copy_task.py

The task
--------
The dataset is generated on the fly: each example is a random sequence of
small integers, and the model's job is to reproduce that same sequence as
its output. Vocab and shape:

    PAD_ID=0, BOS_ID=1, content tokens in [2, 20), seq_len=10

For every batch we construct:

    src    = [c1, c2, ..., c10]               # raw content
    tgt_in = [BOS, c1, c2, ..., c9]           # decoder input (teacher forcing)
    tgt_out = [c1, c2, ..., c10]              # what we want it to predict

This is the same task used by Harvard's Annotated Transformer. It's a
trivially-solvable problem (a perfect copy attention would do it), but
solving it requires every component of the model to work — embedding,
positional encoding, masked self-attention, cross-attention, output
projection. It's the simplest end-to-end check you can run in a few minutes.

What success looks like
-----------------------
After ~10 epochs of 50 batches each (≈10 seconds on M4/MPS):

  * Loss starts around 4.0 (close to log(20) ≈ 3.0 for a uniform predictor,
    plus the label-smoothing offset) and drops below 0.2.
  * Greedy decoding produces sequences that match the source on most or
    all of the test cases.

If your loss doesn't go down, the model isn't learning — start by
inspecting attention masks and the masked self-attention path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import torch
from torch import Tensor

from transformer import ModelConfig, get_device
from transformer.label_smoothing import LabelSmoothingLoss
from transformer.masking import make_pad_mask, make_tgt_mask
from transformer.optim import NoamScheduler
from transformer.transformer import Transformer

# ---------------- Task / dataset constants ----------------------------------
# These define the toy "language" the model has to learn.

VOCAB_SIZE = 20   # total vocab size including PAD (0), BOS (1), and content [2, 20)
PAD_ID = 0        # padding token; never appears in this synthetic task but the
                  # plumbing supports it.
BOS_ID = 1        # "begin of sequence" — prepended to the decoder input so
                  # position 0 has something to attend to before it sees real tokens.
SEQ_LEN = 10      # length of every (src, tgt) sequence
BATCH_SIZE = 32   # examples per training step


def make_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Generate one random copy-task batch.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Length of each source sequence (and content portion of the
            target sequence). Total target length is ``seq_len + 1`` after
            we prepend BOS.
        vocab_size: Upper bound for sampled token IDs. Tokens are drawn from
            [2, vocab_size), avoiding the reserved PAD and BOS IDs.
        device: Where to allocate the tensors. We build them directly on the
            target device to avoid a host→device copy each step.

    Returns:
        src: ``(B, seq_len)`` long tensor of random tokens. The "input"
            sequence the encoder will see.
        tgt: ``(B, seq_len + 1)`` long tensor of ``[BOS, src...]``. Used as
            the decoder input. The training loop will split this into:
                tgt_in  = tgt[:, :-1]   # what the decoder receives
                tgt_out = tgt[:, 1:]    # what we want it to predict
            i.e. classic teacher-forcing.
    """
    src = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    bos = torch.full((batch_size, 1), BOS_ID, device=device, dtype=src.dtype)
    tgt = torch.cat([bos, src], dim=1)
    return src, tgt


def data_iterator(
    n_batches: int, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield ``n_batches`` random copy-task batches.

    A trivial wrapper around ``make_batch``. Uses ``yield`` so the batches
    are produced lazily (we don't allocate all of them up front).
    """
    for _ in range(n_batches):
        yield make_batch(batch_size, seq_len, vocab_size, device)


def train(epochs: int = 10, batches_per_epoch: int = 50) -> Transformer:
    """Train a Transformer on the toy copy task and return the trained model.

    Args:
        epochs: Number of training epochs. Each epoch runs
            ``batches_per_epoch`` random batches and prints a summary.
        batches_per_epoch: Number of optimiser steps per epoch. With the
            defaults (10 epochs × 50 batches) we do ~500 steps total, which
            is plenty for this task.

    Returns:
        The trained model — the caller can then run greedy decoding on it.
    """
    device = get_device()
    # Toy-task config. Smaller than the defaults in ModelConfig because the
    # task is small and we want fast iterations.
    cfg = ModelConfig(
        src_vocab=VOCAB_SIZE,
        tgt_vocab=VOCAB_SIZE,
        d_model=128,        # half of ModelConfig's default; plenty for vocab=20
        n_heads=4,
        n_layers=2,         # 2 layers each side is enough to solve the copy task
        d_ff=512,
        dropout=0.1,
        max_len=SEQ_LEN + 2,  # +2 for safety (BOS at the front, room to spare)
        pad_id=PAD_ID,
    )
    print(f"device={device}")
    print(f"cfg=d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers}")

    # Build, move to device, initialise.
    model = Transformer(cfg).to(device)
    model.init_parameters()

    # Loss: label-smoothed KL divergence with ε=0.1 (paper convention).
    criterion = LabelSmoothingLoss(cfg.tgt_vocab, pad_id=PAD_ID, smoothing=0.1).to(device)

    # Optimiser: Adam with the paper's β values. We start lr=0 and let the
    # NoamScheduler set it on every step (otherwise the first step would
    # use lr=0 anyway, but explicit is better).
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    # Warmup of 400 steps fits well within our 500-step budget.
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=400)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    for epoch in range(1, epochs + 1):
        model.train()        # enables dropout
        total_loss = 0.0     # running token-weighted loss for this epoch
        total_tokens = 0     # running count of non-pad target tokens
        t0 = time.time()

        for src, tgt in data_iterator(batches_per_epoch, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device):
            # Teacher-forcing split:
            #   tgt_in  = [BOS, x1, ..., x_{n-1}]   -- decoder input
            #   tgt_out = [x1, x2, ...,    xn]      -- what we want predicted
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            # Build the masks fresh for each batch — they depend on the
            # actual contents of src/tgt_in (specifically, where the pad
            # tokens are). Cheap to recompute.
            src_mask = make_pad_mask(src, PAD_ID)
            tgt_mask = make_tgt_mask(tgt_in, PAD_ID)

            # Forward pass: (B, S_tgt, V) raw logits.
            logits = model(src, tgt_in, src_mask, tgt_mask)

            # Count non-pad target tokens — used both for loss normalisation
            # and for the running mean we print at end-of-epoch.
            n_tok = (tgt_out != PAD_ID).sum().item()

            # Flatten (B, S, V) → (B*S, V) and (B, S) → (B*S) for the loss.
            # Then divide by token count so the magnitude is comparable
            # regardless of batch / seq_len.
            loss = criterion(
                logits.reshape(-1, cfg.tgt_vocab),
                tgt_out.reshape(-1),
            ) / max(n_tok, 1)

            # Standard step: zero, backward, step, scheduler.step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track the per-token loss so the printed average matches what
            # we'd report on a full epoch. ``loss.item()`` is already the
            # mean over tokens for this batch; multiply back to get the sum.
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        avg_loss = total_loss / max(total_tokens, 1)
        dt = time.time() - t0
        print(
            f"epoch {epoch:2d}  loss={avg_loss:.4f}  "
            f"lr={scheduler.last_lr:.5f}  {batches_per_epoch / dt:.1f} batch/s"
        )

    return model


def greedy_decode(model: Transformer, src: Tensor, max_len: int) -> Tensor:
    """Inference-time decoding: emit one token at a time, picking the argmax.

    Greedy is the simplest possible decoding strategy (vs. beam search,
    sampling, etc.). For the copy task it's enough; for real translation
    you'd want beam search.

    Args:
        model: A trained Transformer.
        src: ``(B, S_src)`` source token IDs.
        max_len: Number of tokens to generate (after BOS, which we strip).

    Returns:
        ``(B, max_len)`` long tensor of generated token IDs.
    """
    model.eval()
    device = src.device
    src_mask = make_pad_mask(src, PAD_ID)

    # Encode the source ONCE — its representation doesn't change as we
    # generate target tokens, so encoding inside the loop would be pure waste.
    memory = model.encode(src, src_mask)

    # Start the decoder with just BOS for every example in the batch.
    B = src.size(0)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

    # ``no_grad`` because we're not training and don't want to build the
    # autograd graph. Saves memory and a tiny bit of compute.
    with torch.no_grad():
        for _ in range(max_len):
            # Build the causal mask for the running ys (it grows by 1 each step).
            tgt_mask = make_tgt_mask(ys, PAD_ID)
            # Run the decoder against the cached memory.
            out = model.decode(memory, ys, src_mask, tgt_mask)
            # We only need the prediction at the LAST position — that's the
            # next token we want to emit. ``out[:, -1]`` slices off
            # (B, d_model). Then project to vocab logits and argmax.
            logits = model.generator(out[:, -1])  # (B, V)
            next_tok = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            # Append the new token to ys and loop.
            ys = torch.cat([ys, next_tok], dim=1)

    # Drop the BOS we started with — return only the generated content.
    return ys[:, 1:]


def evaluate(model: Transformer, n_samples: int = 4) -> None:
    """Generate a few examples and print src/pred side-by-side."""
    device = get_device()
    src = torch.randint(2, VOCAB_SIZE, (n_samples, SEQ_LEN), device=device)
    pred = greedy_decode(model, src, SEQ_LEN)
    # "Exactly correct" = every position matches.
    correct = (pred == src).all(dim=-1).sum().item()
    print(f"\neval: {correct}/{n_samples} sequences copied exactly")
    for i in range(n_samples):
        print(f"  src : {src[i].tolist()}")
        print(f"  pred: {pred[i].tolist()}")


if __name__ == "__main__":
    model = train(epochs=10, batches_per_epoch=50)
    evaluate(model)
