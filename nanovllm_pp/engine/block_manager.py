import math
import json


def compute_block_hash(parent_hash, block_tokens):
    """Hash a KV block based on its parent block and local token IDs.

    When two sequences share a token prefix, their corresponding blocks
    produce identical hashes, enabling prefix-aware KV cache reuse.
    """
    return hash((parent_hash, tuple(block_tokens)))


class Block:
    """A KV cache page — a fixed-size chunk of per-layer key-value memory.

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
    """Paged KV cache memory manager.

    Manages a pool of fixed-size KV cache blocks. Provides:
      - Logical-to-physical address translation via per-sequence block tables.
      - Reference counting for safe block sharing.
      - Chained hashing for prefix cache detection.
      - Copy-on-write semantics when a cached block is partially reused.

    Memory is allocated as a pool of `num_blocks` blocks, each holding
    `block_size` tokens. The allocator maintains free/used sets and a
    hash-to-block mapping for prefix cache lookups.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = set(range(num_blocks))
        self.used_block_ids = set()
        self.hash_to_block_id: dict = {}

    def required_blocks(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, seq, num_new_tokens: int = None) -> bool:
        tokens = seq.prompt_len if num_new_tokens is None else num_new_tokens
        need = self.required_blocks(tokens)
        return len(self.free_block_ids) >= need

    def can_append(self, seq) -> bool:
        need = self.required_blocks(len(seq.token_ids))
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
        block.ref_count = 1
        return block_id

    def allocate(self, seq):
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

            cached_block_id = self.hash_to_block_id.get(block_hash)
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

    def may_append(self, seq) -> bool:
        """Ensure the sequence has a physical block for its next generated token.

        Returns True if allocation succeeded or was not needed.
        """
        needed = self.required_blocks(len(seq.token_ids))
        if needed > len(seq.block_table):
            if not self.free_block_ids:
                return False
            fresh_id = self.allocate_fresh_block()
            seq.block_table.append(fresh_id)
        return True

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
