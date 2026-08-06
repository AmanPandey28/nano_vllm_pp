import hashlib
import json
import math


def compute_block_hash(parent_hash, block_tokens):
    """Hash a KV block based on its parent block and local token IDs.

    When two sequences share a token prefix, their corresponding blocks
    produce identical hashes, enabling prefix-aware KV cache reuse.
    """
    digest = hashlib.sha256()
    digest.update((parent_hash or "ROOT").encode("ascii"))
    for token_id in block_tokens:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


class Block:
    """Logical page metadata for a fixed-size chunk of token positions.

    Reference counting tracks how many sequences share this block.
    When ref_count reaches zero, the block returns to the free pool.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = None
        self.token_ids = []
        # Token metadata can be known before the model has produced K/V for
        # those positions (for example, immediately after prompt admission).
        # Prefix reuse is safe only when the entire physical page is valid.
        self.computed_tokens = 0

    def __repr__(self):
        return (
            f"Block(id={self.block_id}, refs={self.ref_count}, "
            f"computed={self.computed_tokens}, hash={self.hash})"
        )


class BlockManager:
    """Physical-page allocator and admission-control model.

    Manages a pool of fixed-size KV cache blocks. Provides:
      - Logical-to-physical address translation via per-sequence block tables.
      - Reference counting for safe block sharing.
      - Chained hashing for prefix cache detection.

    The manager owns page *metadata* while the paged model runner owns the K/V
    tensors indexed by these page IDs. ``computed_tokens`` is the contract
    between them: a hash is published for reuse only after the corresponding
    physical K/V page has been fully written.
    """

    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = set(range(num_blocks))
        self.used_block_ids = set()
        self.hash_to_block_id: dict = {}

    def required_blocks(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, seq, num_new_tokens: int = 0) -> bool:
        """Return whether a prompt and its immediate output slots fit.

        ``num_new_tokens`` is additional to the prompt length.  In particular,
        the scheduler asks for one extra slot because the final prefill forward
        also samples the first generated token.
        """
        tokens = seq.prompt_len + max(0, num_new_tokens)
        need = self.required_blocks(tokens)
        reusable = self._reusable_prefix_block_ids(seq)
        # A zero-reference cached page is in the free set, so claiming it still
        # consumes one free page. Only an already-active shared page reduces
        # the number of free pages required by this admission.
        active_shared = sum(
            self.blocks[block_id].ref_count > 0 for block_id in reusable
        )
        return len(self.free_block_ids) >= need - active_shared

    def can_append(self, seq, num_new_tokens: int = 1) -> bool:
        need = self.required_blocks(len(seq.token_ids) + max(0, num_new_tokens))
        have = len(seq.block_table)
        if need <= have:
            return True
        return len(self.free_block_ids) >= (need - have)

    def allocate_fresh_block(self):
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks.")
        block_id = self.free_block_ids.pop()
        self.used_block_ids.add(block_id)
        block = self.blocks[block_id]
        # A zero-ref cached block is eligible for reuse.  Evict its old hash
        # before overwriting it so hash_to_block_id can never point at unrelated
        # contents.
        if block.hash is not None and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.ref_count = 1
        block.hash = None
        block.token_ids = []
        block.computed_tokens = 0
        return block_id

    def _reusable_prefix_block_ids(self, seq) -> list[int]:
        """Return the longest physically valid full-block prompt prefix.

        The final logical prompt block is deliberately not reused when the
        prompt ends exactly at a block boundary. At least one prompt token must
        run through the model so the engine obtains logits for the first output
        token without storing per-prefix logits in the cache.
        """
        if not self.enable_prefix_caching:
            return []

        prompt_tokens = seq.token_ids[: seq.prompt_len]
        full_blocks = len(prompt_tokens) // self.block_size
        prompt_blocks = self.required_blocks(len(prompt_tokens))
        max_reusable = min(full_blocks, max(0, prompt_blocks - 1))
        reusable = []
        parent_hash = None
        for logical_block in range(max_reusable):
            start = logical_block * self.block_size
            block_tokens = prompt_tokens[start : start + self.block_size]
            block_hash = compute_block_hash(parent_hash, block_tokens)
            block_id = self.hash_to_block_id.get(block_hash)
            if block_id is None:
                break
            block = self.blocks[block_id]
            if (
                block.hash != block_hash
                or block.token_ids != list(block_tokens)
                or block.computed_tokens != self.block_size
            ):
                break
            reusable.append(block_id)
            parent_hash = block_hash
        return reusable

    def cached_prefix_tokens(self, seq) -> int:
        """Number of prompt tokens that can skip model computation now."""
        return len(self._reusable_prefix_block_ids(seq)) * self.block_size

    def allocate(self, seq, num_new_tokens: int = 0):
        """Allocate blocks for an entire prompt.

        Attempts prefix cache matching on full blocks first. If a cached
        block is found, its ref_count is incremented and the block is
        shared. Otherwise a fresh block is allocated and its hash is
        registered for future reuse.
        """
        if seq.block_table:
            raise RuntimeError("cannot allocate an already-allocated sequence")
        if not self.can_allocate(seq, num_new_tokens):
            raise RuntimeError("Out of KV cache blocks while admitting prompt.")

        prompt_tokens = seq.token_ids[: seq.prompt_len]
        reusable = self._reusable_prefix_block_ids(seq)

        # Claim reusable pages before fresh allocation. Otherwise a free cached
        # page could be selected and evicted by allocate_fresh_block first.
        for block_id in reusable:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            block.ref_count += 1
            seq.block_table.append(block_id)

        seq.num_cached_tokens = len(reusable) * self.block_size
        total_blocks = self.required_blocks(seq.prompt_len + max(0, num_new_tokens))
        while len(seq.block_table) < total_blocks:
            logical_block = len(seq.block_table)
            fresh_id = self.allocate_fresh_block()
            block = self.blocks[fresh_id]
            start = logical_block * self.block_size
            block.token_ids = list(prompt_tokens[start : start + self.block_size])
            if len(block.token_ids) == self.block_size:
                parent_hash = None
                if logical_block > 0:
                    parent_id = seq.block_table[logical_block - 1]
                    parent_hash = self.blocks[parent_id].hash
                block.hash = compute_block_hash(parent_hash, block.token_ids)
            seq.block_table.append(fresh_id)

    def may_append(self, seq, num_new_tokens: int = 1) -> bool:
        """Reserve logical blocks for upcoming generated token positions.

        Returns True if allocation succeeded or was not needed.
        """
        needed = self.required_blocks(len(seq.token_ids) + max(0, num_new_tokens))
        while needed > len(seq.block_table):
            if not self.free_block_ids:
                return False
            fresh_id = self.allocate_fresh_block()
            seq.block_table.append(fresh_id)
        return True

    def physical_block_id(self, seq, token_position: int) -> int:
        """Translate one logical token position through ``seq.block_table``."""
        if token_position < 0:
            raise ValueError("token_position must be non-negative")
        logical_block = token_position // self.block_size
        if logical_block >= len(seq.block_table):
            raise RuntimeError(
                f"token position {token_position} has no reserved KV block "
                f"(table has {len(seq.block_table)} entries)"
            )
        return seq.block_table[logical_block]

    def slot_mapping(self, seq, token_positions) -> list[int]:
        """Map logical token positions to flat physical cache-slot numbers.

        Production cache-write kernels consume the same information, commonly
        as an integer tensor.  This teaching implementation returns Python
        integers because stock Hugging Face attention does not accept a slot
        mapping.
        """
        slots = []
        for position in token_positions:
            position = int(position)
            block_id = self.physical_block_id(seq, position)
            slots.append(block_id * self.block_size + position % self.block_size)
        return slots

    def record_tokens(self, seq, start_position: int, token_ids) -> None:
        """Synchronize token metadata with already-reserved block-table slots.

        The paged runner separately writes K/V tensors to the translated slots.
        Metadata writes must be contiguous and may not mutate a shared prefix.
        """
        for index, token_id in enumerate(token_ids):
            position = start_position + index
            block_id = self.physical_block_id(seq, position)
            block = self.blocks[block_id]
            block_offset = position % self.block_size

            if block_offset < len(block.token_ids):
                if block.token_ids[block_offset] != int(token_id):
                    raise RuntimeError("attempted to overwrite a populated KV slot")
                continue
            if block_offset != len(block.token_ids):
                raise RuntimeError("KV block metadata writes must be contiguous")
            if block.ref_count > 1:
                raise RuntimeError(
                    "copy-on-write is required before mutating a shared block"
                )

            block.token_ids.append(int(token_id))
            if len(block.token_ids) == self.block_size:
                logical_block = position // self.block_size
                parent_hash = None
                if logical_block > 0:
                    parent_id = seq.block_table[logical_block - 1]
                    parent_hash = self.blocks[parent_id].hash
                block.hash = compute_block_hash(parent_hash, block.token_ids)

    def mark_computed(self, seq, start_position: int, num_tokens: int) -> None:
        """Publish that model K/V writes for a contiguous range completed.

        Call this only after every transformer layer has successfully enqueued
        its key/value write on the runner's CUDA stream. Later reads use that
        same stream, so CUDA ordering makes them observe the writes without a
        host synchronization. A full page then becomes eligible for reuse.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        for position in range(start_position, start_position + num_tokens):
            block_id = self.physical_block_id(seq, position)
            block = self.blocks[block_id]
            offset = position % self.block_size
            if offset < block.computed_tokens:
                continue
            if offset != block.computed_tokens:
                raise RuntimeError("physical KV writes must be marked contiguous")
            if offset >= len(block.token_ids):
                raise RuntimeError("cannot mark KV computed before recording token")
            block.computed_tokens += 1
            if (
                self.enable_prefix_caching
                and block.computed_tokens == self.block_size
                and block.hash is not None
            ):
                self.hash_to_block_id.setdefault(block.hash, block_id)

    def record_token(self, seq, token_id: int) -> None:
        """Record the sequence's just-appended token in its physical slot."""
        position = len(seq.token_ids) - 1
        if position < 0 or seq.token_ids[position] != token_id:
            raise RuntimeError("record_token expects the token to be appended first")
        self.record_tokens(seq, position, [token_id])

    def deallocate(self, seq):
        """Release all blocks owned by this sequence.

        Decrements reference counts. Blocks with ref_count == 0 are
        returned to the free pool but retain their hash entries for
        potential future prefix cache reuse.
        """
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.used_block_ids.discard(block_id)
                self.free_block_ids.add(block_id)
        seq.block_table.clear()

    def hash_prompt_blocks(self, token_ids):
        """Compute chained hashes for all full blocks in a token list."""
        results = []
        parent = None
        for i in range(0, len(token_ids), self.block_size):
            block_tokens = token_ids[i : i + self.block_size]
            if len(block_tokens) < self.block_size:
                break
            h = compute_block_hash(parent, block_tokens)
            results.append((h, block_tokens))
            parent = h
        return results

    @property
    def num_free_blocks(self):
        return len(self.free_block_ids)

    def stats(self):
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": len(self.used_block_ids),
            "cached_hashes": len(self.hash_to_block_id),
        }

    def debug_state(self):
        return json.dumps(self.stats(), indent=2)
