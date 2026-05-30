"""Import smoke tests.

Two purposes:

  1. Guarantee that the package is at least syntactically valid and every
     submodule loads cleanly. If any file has an import-time error (bad
     syntax, circular import, missing dependency), one of these tests
     fails immediately — much easier to debug than a confusing failure
     deep in another test.

  2. Sanity-check the public API: ``ModelConfig`` and ``get_device`` are
     supposed to be importable directly from the top-level package, and
     ``ModelConfig`` should compute ``d_k`` correctly.
"""

from __future__ import annotations


def test_package_imports() -> None:
    """Top-level public API works: ModelConfig + get_device."""
    import transformer
    from transformer import ModelConfig, get_device

    cfg = ModelConfig(src_vocab=10, tgt_vocab=10)
    # d_k should always equal d_model // n_heads.
    assert cfg.d_k == cfg.d_model // cfg.n_heads
    # We expect MPS on a Mac and CPU on CI; both are valid.
    assert get_device().type in {"mps", "cpu"}
    assert hasattr(transformer, "ModelConfig")


def test_submodules_importable() -> None:
    """Every module file should at least import cleanly.

    If you add a new module, add it to this list so a broken import is
    caught here rather than the first time someone tries to use it.
    """
    from transformer import (  # noqa: F401  (intentionally unused — import-only test)
        attention,
        decoder,
        encoder,
        feedforward,
        label_smoothing,
        layernorm,
        layers,
        masking,
        optim,
        positional,
        transformer,
    )
