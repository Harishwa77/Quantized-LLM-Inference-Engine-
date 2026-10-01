"""Physical block pool: allocation, refcounts, and free-list management."""

from __future__ import annotations

import torch

from llmopt.utils.logging import get_logger

__all__ = ["BlockAllocator", "BlockPool"]

logger = get_logger("cache.allocator")


class BlockAllocator:
    """A free-list over ``0..num_blocks-1`` backed by a boolean mask.

    Keeping occupancy in a tensor rather than a Python list means the allocator
    is cheap to reset and can be inspected (and asserted on) from tests.
    """

    def __init__(self, num_blocks: int, device: torch.device | str = "cpu") -> None:
        self.num_blocks = num_blocks
        self.device = torch.device(device)
        self.used = torch.zeros(num_blocks, dtype=torch.bool, device=self.device)

    @property
    def num_free(self) -> int:
        return int((~self.used).sum().item())

    @property
    def num_used(self) -> int:
        return int(self.used.sum().item())

    def can_allocate(self, count: int) -> bool:
        return self.num_free >= count

    def allocate(self, count: int) -> list[int]:
        """Pop ``count`` block ids off the free list, lowest first."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if not self.can_allocate(count):
            raise RuntimeError(
                f"cannot allocate {count} blocks, only {self.num_free} free "
                f"of {self.num_blocks}"
            )
        free = (~self.used).nonzero(as_tuple=False).flatten()[:count]
        ids = [int(i) for i in free.tolist()]
        self.used[free] = True
        return ids

    def free(self, block_ids: list[int] | tuple[int, ...]) -> None:
        for block in block_ids:
            if block < 0 or block >= self.num_blocks:
                raise IndexError(f"block id {block} out of range")
            self.used[block] = False

    def reset(self) -> None:
        self.used.zero_()

    def __repr__(self) -> str:
        return f"BlockAllocator(used={self.num_used}, free={self.num_free})"


class BlockPool:
    """Per-sequence block chains with optional shared (prefix-cached) blocks.

    The pool tracks, for every sequence, the ordered list of physical blocks
    that hold its KV cache. Sequences may share a block when they have an
    identical prefix; sharing is safe because a full block is immutable --
    subsequent appends always go to a freshly allocated block.

    Args:
        num_blocks: Physical block capacity.
        block_size: Tokens per block.
        device: Device for the refcount tensor.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks, device)
        self.device = torch.device(device)
        self.chains: dict[int, list[int]] = {}
        self.tokens: dict[int, int] = {}
        self.ref_counts: dict[int, int] = {}
        self.shared: dict[int, int] = {}
        self.num_evictions = 0

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    def tokens_capacity(self) -> int:
        return self.allocator.num_free * self.block_size

    def tokens_used(self, seq_id: int) -> int:
        """Tokens actually written (or reserved) for ``seq_id``."""
        return self.tokens.get(seq_id, 0)

    def can_grow(self, seq_id: int, num_tokens: int) -> bool:
        """Whether appending ``num_tokens`` keeps the sequence within capacity."""
        return self.allocator.can_allocate(self._blocks_needed(seq_id, num_tokens))

    def blocks_needed(self, seq_id: int, num_tokens: int) -> int:
        """How many more blocks ``num_tokens`` additional tokens would require."""
        return self._blocks_needed(seq_id, num_tokens)

    def _blocks_needed(self, seq_id: int, num_tokens: int) -> int:
        """How many extra blocks ``num_tokens`` more tokens require.

        The chain length alone is not enough: the tail block is usually only
        partially filled, so growth is driven by the token count and not by
        ``len(chain) * block_size``.
        """
        current_blocks = len(self.chains.get(seq_id, ()))
        current_tokens = self.tokens.get(seq_id, 0)
        total = current_tokens + num_tokens
        return max(0, -(-total // self.block_size) - current_blocks)

    def append(self, seq_id: int, num_tokens: int, reusable: list[int] | None = None) -> list[int]:
        """Extend ``seq_id``'s chain, returning the newly added block ids.

        Args:
            seq_id: Sequence being extended.
            num_tokens: Number of tokens about to be written.
            reusable: Pre-existing blocks that can be adopted instead of
                allocated (prefix cache hits).
        """
        chain = self.chains.setdefault(seq_id, [])
        needed = self._blocks_needed(seq_id, num_tokens)
        self.tokens[seq_id] = self.tokens.get(seq_id, 0) + num_tokens
        if needed == 0:
            return []
        new_blocks: list[int] = []
        if reusable:
            new_blocks.extend(reusable[:needed])
        if len(new_blocks) < needed:
            new_blocks.extend(self.allocator.allocate(needed - len(new_blocks)))
        for block in new_blocks:
            self.ref_counts[block] = self.ref_counts.get(block, 0) + 1
        chain.extend(new_blocks)
        return new_blocks

    def release(self, seq_id: int) -> list[int]:
        """Drop ``seq_id``'s chain, returning blocks whose refcount hit zero."""
        chain = self.chains.pop(seq_id, [])
        self.tokens.pop(seq_id, None)
        freed: list[int] = []
        for block in chain:
            count = self.ref_counts.get(block, 0) - 1
            if count <= 0:
                self.ref_counts.pop(block, None)
                self.shared.pop(block, None)
                freed.append(block)
            else:
                self.ref_counts[block] = count
        if freed:
            self.allocator.free(freed)
        return freed

    def share_blocks(self, seq_id: int, block_ids: list[int], num_tokens: int | None = None) -> None:
        """Record that ``block_ids`` are also referenced by ``seq_id``."""
        chain = self.chains.setdefault(seq_id, [])
        for block in block_ids:
            self.ref_counts[block] = self.ref_counts.get(block, 0) + 1
            chain.append(block)
        shared_tokens = len(block_ids) * self.block_size
        self.tokens[seq_id] = max(self.tokens.get(seq_id, 0), num_tokens or shared_tokens)

    def release_block(self, block_id: int) -> None:
        """Drop the prefix cache's reference to ``block_id``.

        Only the cache calls this, so the reference being given up is always the
        cache's own. The ``shared`` marker must be cleared too: leaving it behind
        would make a later :meth:`mark_shared` treat the block as still pinned
        and skip taking a reference for it, letting the allocator recycle a block
        the cache still addresses.

        Blocks still referenced by live sequences are left alone, because their
        refcount has not reached zero.
        """
        self.shared.pop(block_id, None)
        count = self.ref_counts.get(block_id, 0)
        if count > 1:
            self.ref_counts[block_id] = count - 1
            return
        self.ref_counts.pop(block_id, None)
        for seq_id, chain in list(self.chains.items()):
            if block_id in chain:
                chain.remove(block_id)
                self.tokens[seq_id] = min(
                    self.tokens.get(seq_id, 0), len(chain) * self.block_size
                )
        self.allocator.free([block_id])
        self.num_evictions += 1

    def mark_shared(self, block_ids: list[int]) -> None:
        """Pin ``block_ids`` on behalf of the prefix cache.

        The cache keeps addressing these blocks by id long after the sequence
        that filled them is released or preempted, so it has to hold a reference
        of its own. Without one the allocator recycles the block and a later
        cache hit silently hands a sequence the KV of a different one.

        Idempotent: the cache owns exactly one reference per block, so a block
        that is evicted and later re-published (after a preemption cycle) is
        not pinned twice.
        """
        for block in block_ids:
            if block in self.shared:
                continue
            self.shared[block] = 1
            self.ref_counts[block] = self.ref_counts.get(block, 0) + 1

    def is_shared(self, block_id: int) -> bool:
        return self.ref_counts.get(block_id, 0) > 1

    def stats(self) -> dict[str, int]:
        return {
            "num_blocks": self.allocator.num_blocks,
            "num_used": self.allocator.num_used,
            "num_free": self.allocator.num_free,
            "num_sequences": len(self.chains),
            "num_shared": len(self.shared),
            "num_evictions": self.num_evictions,
        }

    def reset(self) -> None:
        self.allocator.reset()
        self.chains.clear()
        self.tokens.clear()
        self.ref_counts.clear()
        self.shared.clear()

    def __repr__(self) -> str:
        return f"BlockPool(block_size={self.block_size}, {self.stats()})"
