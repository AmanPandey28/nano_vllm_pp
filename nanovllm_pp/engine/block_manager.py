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

    def __repr__(self):
        return f"Block(id={self.block_id}, refs={self.ref_count}, hash={self.hash})"


class BlockManager:
    """Logical page allocator and admission-control model.

    Manages a pool of fixed-size KV cache blocks. Provides:
      - Logical-to-physical address translation via per-sequence block tables.
      - Reference counting for safe block sharing.
      - Chained hashing for prefix cache detection.

    Important: these objects contain token metadata, not K/V tensors. The
    executable model path stores K/V in each sequence's Hugging Face cache.
    Consequently prefix sharing is disabled at the engine boundary until a
    physical paged-attention backend is integrated. This allocator still makes
    capacity, lifecycle, lookahead, and block-table behavior inspectable.
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
        return len(self.free_block_ids) >= need

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
        return block_id

    def allocate(self, seq, num_new_tokens: int = 0):
        """Allocate blocks for an entire prompt.

        Attempts prefix cache matching on full blocks first. If a cached
        block is found, its ref_count is incremented and the block is
        shared. Otherwise a fresh block is allocated and its hash is
        registered for future reuse.
        """
        prompt_tokens = seq.token_ids[: seq.prompt_len]
        num_full_blocks = len(prompt_tokens) // self.block_size

        for i in range(num_full_blocks):
            block_tokens = prompt_tokens[
                i * self.block_size : (i + 1) * self.block_size
            ]
            parent_hash = (
                self.blocks[seq.block_table[-1]].hash if seq.block_table else None
            )
            block_hash = compute_block_hash(parent_hash, block_tokens)

            cached_block_id = (
                self.hash_to_block_id.get(block_hash)
                if self.enable_prefix_caching
                else None
            )
            if cached_block_id is not None:
                cached_block = self.blocks[cached_block_id]
                if cached_block.ref_count == 0:
                    cached_block.ref_count = 1
                    self.free_block_ids.discard(cached_block_id)
                    self.used_block_ids.add(cached_block_id)
                else:
                    cached_block.ref_count += 1
                seq.block_table.append(cached_block_id)
                seq.num_cached_tokens += self.block_size
            else:
                fresh_id = self.allocate_fresh_block()
                fresh_block = self.blocks[fresh_id]
                fresh_block.hash = block_hash
                fresh_block.token_ids = list(block_tokens)
                if self.enable_prefix_caching:
                    self.hash_to_block_id[block_hash] = fresh_id
                seq.block_table.append(fresh_id)

        remaining = len(prompt_tokens) % self.block_size
        if remaining > 0:
            fresh_id = self.allocate_fresh_block()
            fresh_block = self.blocks[fresh_id]
            fresh_block.token_ids = list(
                prompt_tokens[num_full_blocks * self.block_size :]
            )
            seq.block_table.append(fresh_id)

        if num_new_tokens and not self.may_append(seq, num_new_tokens):
            # Callers should use can_allocate first.  Keep this guard so direct
            # use cannot leave a silently undersized block table.
            self.deallocate(seq)
            raise RuntimeError("Out of KV cache blocks while reserving lookahead.")

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

        K/V tensors remain owned by the Hugging Face cache, but maintaining the
        control-plane contents makes address translation and block lifecycle
        independently testable.  Writes must be contiguous and may not mutate a
        shared prefix block.
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
                if self.enable_prefix_caching:
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
