from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PagedBatchMetadata:
    """Tensorized logical-to-physical mapping for one model forward.

    ``query_start_loc`` partitions flattened input tokens by request.
    ``context_lens`` contains each request's cache length after this forward,
    and each row of ``block_tables`` maps logical pages to physical page IDs.
    """

    positions: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    is_decode: bool
    block_size: int
    query_starts: tuple[int, ...]
    context_lengths: tuple[int, ...]
    max_context_len: int

    @property
    def num_sequences(self) -> int:
        return len(self.context_lengths)
