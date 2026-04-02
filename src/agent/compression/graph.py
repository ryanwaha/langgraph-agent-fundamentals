"""Thin adapter: wires agent.llm model factories into lossless_compressor.

Existing call sites continue to use:
    from agent.compression.graph import run_compression

The actual subgraph lives in the lossless_compressor submodule at
src/lossless_compressor/. This file is the only place that knows about
both agent.llm and lossless_compressor.
"""

from agent.llm import get_compress_model, get_fallback_compress_model
from lossless_compressor import configure, run_compression  # noqa: F401

configure(
    model_factory=get_compress_model,
    fallback_factory=get_fallback_compress_model,
)

__all__ = ["run_compression"]
