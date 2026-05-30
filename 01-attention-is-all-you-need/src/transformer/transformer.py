"""The full Transformer model: tying everything else together.

Paper section: §3 (the entire model), Figure 1 of the paper.

What this file does
-------------------
Up to this point we've built isolated components — embeddings, positional
encoding, attention, FFN, encoder layer, decoder layer, encoder stack,
decoder stack. This file assembles them into the end-to-end model:

    src token IDs  →  src_embed → src_pos → encoder → memory
    tgt token IDs  →  tgt_embed → tgt_pos → decoder ←──┘
                                              ↓
                                         generator (Linear)
                                              ↓
                                       (B, S_tgt, tgt_vocab) logits

Two paper details that are easy to miss
---------------------------------------
1. **Embedding scaling** (paper §3.4). After the token embedding lookup we
   multiply by ``√d_model``. Without this the embeddings would be tiny
   relative to the positional encodings (whose entries are in [-1, 1] but
   sum to roughly the right magnitude), so the position info would dominate.
   Implemented in ``Embeddings.forward``.

2. **Weight tying** (paper §3.4). The output projection (``generator``) shares
   weights with the target token embedding. Same matrix, used in two places:
   to embed input tokens, and to score output tokens. This is parameter-
   efficient (one matrix instead of two with the same shape) and a long-
   established trick in language modelling. Implemented in ``Transformer.__init__``.
"""

from __future__ import annotations

import math

from torch import Tensor, nn

from .config import ModelConfig
from .decoder import Decoder
from .encoder import Encoder
from .positional import PositionalEncoding


class Embeddings(nn.Module):
    """Token embedding scaled by ``√d_model``.

    Wraps a plain ``nn.Embedding`` and applies the paper's scaling factor
    on the way out. Used twice in ``Transformer`` — once for the source
    side, once for the target side.

    Args:
        vocab_size: Number of distinct tokens. The embedding matrix will
            be ``(vocab_size, d_model)``.
        d_model: Embedding dimension. Same as the residual-stream width.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # ``lut`` is short for "lookup table". This is the actual learnable
        # embedding matrix.
        self.lut = nn.Embedding(vocab_size, d_model)
        # Stored on the module so ``forward`` can use it without re-deriving.
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """Look up embeddings and scale.

        Args:
            x: ``(B, S)`` long tensor of token IDs.

        Returns:
            ``(B, S, d_model)`` float tensor.
        """
        # The √d_model multiplier is the paper's eq. just above §3.5.
        return self.lut(x) * math.sqrt(self.d_model)


class Transformer(nn.Module):
    """The end-to-end encoder-decoder Transformer from the paper.

    Construction order (which matters because of weight tying):
      1. Build the two embeddings (src and tgt).
      2. Build the two positional encodings (one per side; they share the
         same formula but are independent ``nn.Module`` instances so they
         can hold separate dropout statistics).
      3. Build the encoder and decoder stacks.
      4. Build the output projection (``generator``) and tie its weight to
         the target embedding's weight.

    Sub-modules:
        src_embed:    ``Embeddings(src_vocab, d_model)``
        tgt_embed:    ``Embeddings(tgt_vocab, d_model)``
        src_pos:      ``PositionalEncoding`` for the source side
        tgt_pos:      ``PositionalEncoding`` for the target side
        encoder:      ``Encoder`` — the N-layer encoder stack
        decoder:      ``Decoder`` — the N-layer decoder stack
        generator:    ``nn.Linear(d_model, tgt_vocab, bias=False)`` — projects
                      the decoder's final state to vocabulary logits. Bias-free
                      because we tie its weight to the target embedding (which
                      itself has no bias).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Stash the config for downstream code that wants it (e.g.
        # scripts/train_copy_task.py reads ``model.cfg`` for max_len).
        self.cfg = cfg

        # ---- Embeddings + positional encoding ----------------------------
        self.src_embed = Embeddings(cfg.src_vocab, cfg.d_model)
        self.tgt_embed = Embeddings(cfg.tgt_vocab, cfg.d_model)
        # Two PE modules — same formula, but separate instances so the
        # dropout RNG state doesn't entangle the two sides. Memory cost is
        # one extra (max_len, d_model) buffer; trivial.
        self.src_pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.tgt_pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        # ---- The two stacks ---------------------------------------------
        self.encoder = Encoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
        self.decoder = Decoder(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)

        # ---- Output projection (tied to target embedding) ----------------
        # ``bias=False`` because the embedding has no bias; if we wanted a
        # bias we'd be tying mismatched objects.
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab, bias=False)
        # The actual weight-tying. Both modules now point to the same
        # ``nn.Parameter`` object; gradients from both call sites accumulate
        # into the same buffer. PyTorch dedupes this in ``.parameters()`` so
        # it isn't double-counted by the optimizer.
        self.generator.weight = self.tgt_embed.lut.weight

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """Run the encoder side: src tokens → memory.

        Pulled out as its own method (rather than buried inside ``forward``)
        because greedy decoding wants to encode the source once and then
        run the decoder repeatedly against the same memory.

        Args:
            src: ``(B, S_src)`` long tensor of source token IDs.
            src_mask: ``(B, 1, 1, S_src)`` source padding mask.

        Returns:
            ``(B, S_src, d_model)`` encoder memory.
        """
        # Embed → add positional info → run the encoder stack.
        return self.encoder(self.src_pos(self.src_embed(src)), src_mask)

    def decode(
        self,
        memory: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the decoder side: memory + partial tgt → decoder state.

        Note this returns the *pre-generator* decoder state, not logits.
        Greedy decoding calls ``model.generator`` separately so it can
        pick out just the last position.

        Args:
            memory: ``(B, S_src, d_model)`` from a previous ``encode``.
            tgt: ``(B, S_tgt)`` decoder input (typically [BOS, y1, ..., y_{n-1}]).
            src_mask, tgt_mask: see ``masking.py``.

        Returns:
            ``(B, S_tgt, d_model)`` final decoder state.
        """
        return self.decoder(self.tgt_pos(self.tgt_embed(tgt)), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Full forward pass: tokens in, logits out.

        Args:
            src: ``(B, S_src)`` source token IDs.
            tgt: ``(B, S_tgt)`` decoder input token IDs.
            src_mask, tgt_mask: see ``masking.py``.

        Returns:
            ``(B, S_tgt, tgt_vocab)`` raw logits over the target vocabulary.
            (Caller takes softmax / argmax / loss as appropriate — we return
            raw scores here so a label-smoothing or cross-entropy loss can
            do the log-softmax itself for numerical stability.)
        """
        memory = self.encode(src, src_mask)
        out = self.decode(memory, tgt, src_mask, tgt_mask)
        # Project decoder state from d_model → tgt_vocab. With weight tying
        # this matmul shares weights with the embedding, so it's effectively
        # asking "how well does this hidden state look like each vocab item's
        # embedding?".
        return self.generator(out)

    def init_parameters(self) -> None:
        """Xavier-initialise every parameter with rank > 1.

        Why rank > 1: we only initialise weight matrices, not biases or
        scalar/vector parameters (LayerNorm gamma/beta, embeddings already
        come from ``nn.Embedding`` defaults, etc.). Xavier uniform on weights
        is a long-standing convention for transformer training and matches
        the Annotated Transformer.

        Call this once after construction (and after ``.to(device)``) before
        starting training.
        """
        for p in self.parameters():
            if p.dim() > 1:
                # Xavier uniform: U(-a, a) with a chosen so the variance of
                # the layer's output matches the variance of its input.
                # Helps avoid the "deeper layers have shrinking activations"
                # problem at initialisation.
                nn.init.xavier_uniform_(p)
