"""Content-addressed prefix cache over full KV blocks.

Each full block is addressed by the hash of the token ids it covers, chained
with the hash of the preceding block, so any two sequences sharing a token
prefix share physical blocks. Only *full* blocks are cached: a partial tail
block can still be appended to, and mutating it would corrupt other sequences
that reference it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from llmopt.utils.logging import get_logger

__all__ = ["CachedPrefix", "PrefixCache", "block_hash"]

logger = get_logger("cache.prefix")


def block_hash(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    """Content-address a block from its parent hash and the tokens it stores."""
    payload = f"{parent_hash}:" + ",".join(str(token) for token in token_ids)
    digest = hashlib.blake2b(payload.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(slots=True)
class CachedPrefix:
    """A reusable KV prefix expressed in complete blocks."""

    num_blocks: int
    block_ids: list[int]
    num_tokens: int

    def __len__(self) -> int:
        return self.num_blocks


@dataclass(slots=True)
class _Node:
    """One trie level: exactly one cached block plus its children."""

    key: int
    block_id: int
    parent: _Node | None
    depth: int
    children: dict[int, _Node] = field(default_factory=dict)

    def walk_up(self) -> list[int]:
        """Block ids from the root down to (and including) this node."""
        chain: list[int] = []
        node: _Node | None = self
        while node is not None and node.block_id >= 0:
            chain.append(node.block_id)
            node = node.parent
        chain.reverse()
        return chain


class PrefixCache:
    """Hash-linked trie of cached blocks with LRU eviction.

    Args:
        block_pool: Pool owning the physical blocks. Must expose
            ``release_block(block_id)`` so eviction can hand memory back.
        block_size: Tokens per block.
        max_entries: Maximum number of cached blocks before LRU eviction.
    """

    def __init__(self, block_pool: Any, block_size: int, max_entries: int = 4096) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_pool = block_pool
        self.block_size = block_size
        self.max_entries = max(1, max_entries)
        self.root = _Node(key=0, block_id=-1, parent=None, depth=0)
        self._nodes: dict[int, _Node] = {}
        self._lru: list[int] = []
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def find(self, token_ids: list[int]) -> CachedPrefix | None:
        """Return the longest cached prefix of ``token_ids`` covering full blocks.

        Returns ``None`` on a complete miss. On a partial hit the returned
        prefix is truncated to a block boundary, which is what the caller needs
        since a partial block cannot be shared.
        """
        node = self.root
        for index in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            chunk = tuple(token_ids[index : index + self.block_size])
            child = node.children.get(block_hash(node.key, chunk))
            if child is None:
                break
            node = child
        if node is self.root:
            self.misses += 1
            return None
        self.hits += 1
        self._touch(node.key)
        return CachedPrefix(
            num_blocks=node.depth,
            block_ids=node.walk_up(),
            num_tokens=node.depth * self.block_size,
        )

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None:
        """Register ``block_ids`` as the cacheable prefix of ``token_ids``."""
        if not block_ids:
            return
        node = self.root
        for index, block_id in enumerate(block_ids):
            start = index * self.block_size
            chunk = tuple(token_ids[start : start + self.block_size])
            if len(chunk) < self.block_size:
                # A partial trailing block is never cached: it can still grow.
                break
            key = block_hash(node.key, chunk)
            child = node.children.get(key)
            if child is None:
                child = _Node(
                    key=key, block_id=block_id, parent=node, depth=node.depth + 1
                )
                node.children[key] = child
                self._nodes[key] = child
            elif child.block_id != block_id:
                # Identical content held in a different physical block: adopt the
                # new block and release the stale one so capacity is not leaked.
                self._release_block(child.block_id)
                child.block_id = block_id
            node = child
            self._touch(key)
        self._enforce_capacity()

    def contains(self, token_ids: list[int]) -> bool:
        return self.find(list(token_ids)) is not None

    def _touch(self, key: int) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _release_block(self, block_id: int) -> None:
        release = getattr(self.block_pool, "release_block", None)
        if callable(release):
            release(block_id)

    def _enforce_capacity(self) -> None:
        while len(self._lru) > self.max_entries:
            self._evict(self._lru.pop(0))

    def evict_until_free(self, num_blocks: int) -> int:
        """Evict least-recently-used cached blocks until ``num_blocks`` are free.

        ``max_entries`` alone is not enough to bound memory: a long run of
        distinct prompts can publish more blocks than the pool can hold, and
        every one of them stays pinned by the cache while the engine starves.
        Eviction is therefore also demand driven -- when an allocation fails the
        scheduler asks for room here, trading cached prefixes (pure recompute
        cost) for blocks (hard capacity).

        Args:
            num_blocks: Number of free blocks the caller needs.

        Returns:
            How many cache entries were evicted.
        """
        free_blocks = getattr(self.block_pool, "num_free_blocks", None)
        if free_blocks is None:
            return 0
        # Tolerate either a property or a method on the pool.
        probe = free_blocks if callable(free_blocks) else lambda: free_blocks
        evicted = 0
        # Bounded by the cache size: evicting a block that other sequences still
        # reference only drops a reference and frees nothing, so stop after the
        # cache has been walked once rather than spinning.
        budget = len(self._lru)
        while probe() < num_blocks and budget > 0:
            budget -= 1
            self._evict(self._lru.pop(0))
            evicted += 1
        return evicted

    def _evict(self, key: int) -> None:
        node = self._nodes.pop(key, None)
        if node is None or node.parent is None:
            return
        node.parent.children.pop(key, None)
        self._release_block(node.block_id)
        # Detach the evicted subtree so its blocks are not orphaned.
        for descendant in self._descendants(node):
            self._nodes.pop(descendant.key, None)
            self._release_block(descendant.block_id)

    def _descendants(self, node: _Node) -> list[_Node]:
        out: list[_Node] = []
        stack = list(node.children.values())
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(current.children.values())
        return out

    def reset(self) -> None:
        """Drop every cached block, returning the pool's memory to the allocator.

        The cache holds a reference on each block it publishes, so clearing the
        trie without releasing those references would strand the blocks for the
        lifetime of the pool. The pool's own set of cache-pinned blocks is used
        as the authority here rather than the trie: a block whose node was
        replaced by an identical one is still pinned, and walking the trie alone
        would leak it.
        """
        pinned = getattr(self.block_pool, "shared", None)
        if isinstance(pinned, dict):
            for block_id in list(pinned):
                self._release_block(block_id)
        self.root = _Node(key=0, block_id=-1, parent=None, depth=0)
        self._nodes.clear()
        self._lru.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, float | int]:
        return {
            "entries": len(self._lru),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"PrefixCache(block_size={self.block_size}, entries={len(self._lru)}, "
            f"hits={self.hits}, misses={self.misses})"
        )
