"""Physical paged-KV execution backend.

The implementation is intentionally small enough to study: PyTorch performs
model layers and reference prefill attention, while Triton owns the two
page-specific operations—scattered K/V writes and page-table decode reads.
"""

from .model import PagedQwen3ForCausalLM, load_paged_qwen3
from .runner import PagedModelRunner

__all__ = ["PagedModelRunner", "PagedQwen3ForCausalLM", "load_paged_qwen3"]
