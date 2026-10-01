"""Block tables and the paged KV-cache tensor store."""

from __future__ import annotations

import torch

__all__ = ["BlockTable", "PagedKVCache"]


class BlockTable:
    """Logical-to-physical block mapping for a batch of sequences.

    A block table row is a list of physical block ids. The KV cache tensor is
    addressed as ``[num_blocks, block_size, num_kv_heads, head_dim]``; the table
    translates a (sequence, token offset) pair into a flat address.

    Args:
        max_num_seqs: Number of rows the table can hold.
        max_blocks_per_seq: Maximum blocks any single sequence may occupy.
        device: Device the int32 index tensor lives on.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_blocks_per_seq: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if max_num_seqs < 1 or max_blocks_per_seq < 1:
            raise ValueError("max_num_seqs and max_blocks_per_seq must be >= 1")
        self.max_num_seqs = max_num_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = torch.device(device)
        self.table = torch.zeros(
            (max_num_seqs, max_blocks_per_seq), dtype=torch.int32, device=self.device
        )
        self.lengths = torch.zeros(max_num_seqs, dtype=torch.int32, device=self.device)
        self.row_of: dict[int, int] = {}
        self.num_used_rows = 0

    def allocate_row(self, seq_id: int) -> int:
        """Assign a free row to ``seq_id`` and return its index."""
        if seq_id in self.row_of:
            return self.row_of[seq_id]
        if self.num_used_rows >= self.max_num_seqs:
            raise RuntimeError("block table is full")
        row = self.num_used_rows
        self.num_used_rows += 1
        self.row_of[seq_id] = row
        self.table[row].zero_()
        self.lengths[row] = 0
        return row

    def free_row(self, seq_id: int) -> None:
        row = self.row_of.pop(seq_id, None)
        if row is None:
            return
        self.table[row].zero_()
        self.lengths[row] = 0
        self.num_used_rows -= 1
        self._compact()

    def _compact(self) -> None:
        """Move the last live row into the hole left by ``free_row``."""
        last = self.num_used_rows
        for seq_id, row in list(self.row_of.items()):
            if row == last:
                continue
            self.table[row] = self.table[last]
            self.lengths[row] = self.lengths[last]
            self.row_of[seq_id] = row
            self.table[last].zero_()
            self.lengths[last] = 0

    def set_blocks(self, seq_id: int, blocks: list[int] | torch.Tensor) -> None:
        row = self.row_of[seq_id]
        ids = torch.as_tensor(blocks, dtype=torch.int32, device=self.device).flatten()
        if ids.numel() > self.max_blocks_per_seq:
            raise ValueError(
                f"sequence {seq_id} needs {ids.numel()} blocks, "
                f"table allows {self.max_blocks_per_seq}"
            )
        self.table[row].zero_()
        self.table[row, : ids.numel()] = ids

    def get_blocks(self, seq_id: int) -> torch.Tensor:
        row = self.row_of[seq_id]
        used = self.lengths[row].item()
        return self.table[row, :used]

    def set_length(self, seq_id: int, num_blocks: int) -> None:
        row = self.row_of[seq_id]
        if num_blocks > self.max_blocks_per_seq:
            raise ValueError("num_blocks exceeds max_blocks_per_seq")
        self.lengths[row] = num_blocks

    def get_length(self, seq_id: int) -> int:
        return int(self.lengths[self.row_of[seq_id]].item())

    def gather(self, seq_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_ids, context_lens)`` for a list of sequences.

        ``block_ids`` is padded to the longest sequence with ``-1`` so it can be
        indexed directly; ``context_lens`` gives the true token count each row
        should contribute.
        """
        rows = [self.row_of[seq_id] for seq_id in seq_ids]
        used = [int(self.lengths[row].item()) for row in rows]
        width = max(used, default=0)
        out = torch.full((len(rows), width), -1, dtype=torch.int32, device=self.device)
        for i, (row, count) in enumerate(zip(rows, used, strict=True)):
            if count:
                out[i, :count] = self.table[row, :count]
        lengths = torch.tensor(
            [count * self._block_size for count in used], dtype=torch.int32, device=self.device
        )
        return out, lengths

    _block_size: int = 1

    def set_block_size(self, block_size: int) -> None:
        self._block_size = block_size

    def to(self, device: torch.device | str) -> BlockTable:
        self.table = self.table.to(device)
        self.lengths = self.lengths.to(device)
        self.device = torch.device(device)
        return self


class PagedKVCache:
    """Fixed-capacity paged key/value cache shared by all layers.

    Memory is preallocated as two tensors of shape
    ``[num_blocks, block_size, num_kv_heads, head_dim]``. Because storage never
    moves, it can be bound directly as a CUDA graph capture target, and the
    allocator never fragments as sequences grow and shrink.

    Args:
        num_layers: Number of decoder layers sharing this allocator.
        num_blocks: Total physical blocks.
        block_size: Tokens per block.
        num_kv_heads: Key/value head count (may be < query heads for GQA).
        head_dim: Per-head dimension.
        dtype: Storage dtype.
        device: Device to allocate on.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.key_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.value_cache = torch.zeros_like(self.key_cache)

    def write(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter a batch of KV into the cache.

        Args:
            layer_idx: Destination layer.
            block_ids: ``[seq, max_blocks]`` block table, ``-1`` padded.
            token_offsets: ``[seq]`` absolute position in the context where each
                sequence's first written token belongs.
            keys: ``[seq, q_len, kv_heads, head_dim]``.
            values: Same shape as ``keys``.

        After a continuous-batching step the sequences have different lengths, so
        offsets are tracked per sequence rather than once for the whole batch.
        """
        seq_len, q_len = keys.shape[0], keys.shape[1]
        if q_len == 0 or seq_len == 0:
            return
        for i in range(seq_len):
            start = int(token_offsets[i].item())
            self._write_one(layer_idx, block_ids[i], start, keys[i], values[i], q_len)

    def _write_one(
        self,
        layer_idx: int,
        blocks: torch.Tensor,
        start: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_len: int,
    ) -> None:
        """Write one sequence's ``q_len`` tokens, splitting at block boundaries."""
        written = 0
        cursor = start
        block_index = start // self.block_size
        while written < q_len and block_index < blocks.numel():
            block = int(blocks[block_index].item())
            if block < 0:
                break
            room = self.block_size - (cursor % self.block_size)
            take = min(room, q_len - written)
            lo = cursor % self.block_size
            self.key_cache[layer_idx, block, lo : lo + take] = keys[written : written + take].to(
                self.dtype
            )
            self.value_cache[layer_idx, block, lo : lo + take] = values[
                written : written + take
            ].to(self.dtype)
            written += take
            cursor += take
            if cursor % self.block_size == 0:
                block_index += 1

    def write_decode(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Vectorised single-token write for the decode path.

        ``keys``/``values`` are ``[seq, kv_heads, head_dim]``. Positions land in
        the tail block at ``offset % block_size``; because the block index and
        slot are both known up front the whole batch scatters in one operation.
        """
        seq_len = keys.shape[0]
        if seq_len == 0:
            return
        offsets = token_offsets.to(torch.long)
        block_index = torch.div(offsets, self.block_size, rounding_mode="floor")
        slot = offsets % self.block_size
        block_ids = block_ids.to(torch.long)
        cols = torch.arange(seq_len, device=self.device)
        blocks = block_ids[cols, block_index]
        valid = (blocks >= 0) & (blocks < self.num_blocks)
        if not bool(valid.all()):
            blocks = torch.where(valid, blocks, torch.zeros_like(blocks))
        self.key_cache[layer_idx, blocks, slot] = keys.to(self.dtype)
        self.value_cache[layer_idx, blocks, slot] = values.to(self.dtype)

    def gather(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble contiguous KV for a batch of ragged sequences.

        Args:
            layer_idx: Which layer to read.
            block_ids: ``[seq, max_blocks]`` padded with ``-1``.
            context_lens: ``[seq]`` true token counts.

        Returns:
            ``(keys, values, valid_mask)`` where keys/values are
            ``[seq, context_len, kv_heads, head_dim]`` padded to the longest
            sequence, and ``valid_mask`` is ``[seq, context_len]`` boolean.
        """
        seq_len = block_ids.shape[0]
        max_len = int(context_lens.max().item()) if seq_len else 0
        if max_len == 0:
            head = (0, self.num_kv_heads, self.head_dim)
            empty = torch.zeros((seq_len, *head), dtype=self.dtype, device=self.device)
            return empty, empty.clone(), torch.zeros((seq_len, 0), dtype=torch.bool, device=self.device)

        positions = torch.arange(max_len, device=self.device)
        block_index = positions // self.block_size
        offset = positions % self.block_size

        ids = block_ids[:, block_index]
        valid = ids >= 0
        safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
        valid &= positions.unsqueeze(0) < context_lens.unsqueeze(1)

        width = self.num_kv_heads * self.head_dim
        # Collapse (block, slot) into one axis and (kv_heads, head_dim) into
        # another, so a single gather along dim 0 retrieves a whole token's KV.
        # Advanced indexing cannot do this: it would append a trailing axis.
        key_view = self.key_cache[layer_idx].reshape(-1, width)
        value_view = self.value_cache[layer_idx].reshape(-1, width)

        token_index = (safe_ids * self.block_size + offset).reshape(-1, 1).expand(-1, width)
        keys = key_view.gather(0, token_index).reshape(seq_len, max_len, width)
        values = value_view.gather(0, token_index).reshape(seq_len, max_len, width)

        keys = keys.reshape(seq_len, max_len, self.num_kv_heads, self.head_dim)
        values = values.reshape(seq_len, max_len, self.num_kv_heads, self.head_dim)
        mask = valid[:, :, None, None]
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        return torch.where(mask, keys, zero), torch.where(mask, values, zero), valid

    def memory_bytes(self) -> int:
        return (
            self.key_cache.numel() * self.key_cache.element_size()
            + self.value_cache.numel() * self.value_cache.element_size()
        )

    def reset(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()

    def __repr__(self) -> str:
        from llmopt.utils.misc import human_bytes

        return (
            f"PagedKVCache(layers={self.num_layers}, blocks={self.num_blocks}, "
            f"block_size={self.block_size}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, size={human_bytes(self.memory_bytes())})"
        )
