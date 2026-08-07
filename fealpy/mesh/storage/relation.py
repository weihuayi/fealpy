from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

from ...backend import bm, Tensor

__all__ = ["Relation"]


class UniqueResult(NamedTuple):
    first: Tensor
    last: Tensor


class LocalIndexResult(NamedTuple):
    floc: Tensor
    lloc: Tensor


@dataclass(frozen=True)
class Relation:
    src_name: str
    tgt_name: str
    tgt_indices: Tensor
    """
    The indices to the target entities. This can be a 2D array for
    homogeneous relations (e.g., edges) or a 1D array for heterogeneous
    relations (e.g., node-to-face).
    """
    src_indices: Tensor | None = None
    """
    The indices to the source entities. This is `None` for homogeneous relations
    (e.g., edges) and a 1D array for heterogeneous relations (e.g., node-to-face).
    """
    def heterogeneous_indices(self, copy: bool = True) -> tuple[Tensor, Tensor]:
        """Return the source and target indices for heterogeneous relations.

        Parameters:
            copy (bool, optional): Whether to copy the indices if already heterogeneous.
                Defaults to True.

        Returns:
            tuple[Tensor, Tensor]: The source and target indices.
        """
        if self.src_indices is None:
            src = bm.arange(self.tgt_indices.shape[0], dtype=bm.int32)
            src = bm.repeat(src, self.tgt_indices.shape[1])
            tgt = bm.reshape(bm.copy(self.tgt_indices), (-1,))
        else:
            if copy:
                src = bm.copy(self.src_indices)
                tgt = bm.copy(self.tgt_indices)
            else:
                src = self.src_indices
                tgt = self.tgt_indices

        return src, tgt

    def as_array(self) -> Tensor:
        if self.src_indices is None:
            return self.tgt_indices
        raise ValueError("Cannot convert heterogeneous relation to array. "
                         "Use `as_coo` or `as_csr` instead.")

    def as_coo(self):
        from ...sparse import coo_matrix
        src, tgt = self.heterogeneous_indices()
        data = bm.ones_like(src, dtype=bm.bool)
        return coo_matrix(
            (data, (src, tgt)),
            shape=(bm.max(src) + 1, bm.max(tgt) + 1) # type: ignore
        )

    def as_csr(self):
        return self.as_coo().tocsr()

    def inverse(self) -> Relation:
        """Create an inverse relation."""
        src, tgt = self.heterogeneous_indices()
        return Relation(
            src_name=self.tgt_name,
            tgt_name=self.src_name,
            src_indices=tgt,
            tgt_indices=src,
        )

    @cached_property
    def _fl_mask(self):
        assert self.src_indices is not None
        arg = bm.argsort(self.src_indices)
        src = self.src_indices[arg]
        TRUE = bm.ones((1,), dtype=bm.bool, device=src.device)
        diff = src[1:] != src[:-1]
        diff0 = bm.concat([TRUE, diff])
        diff1 = bm.concat([diff, TRUE])
        return arg, diff0, diff1

    @cached_property
    def unique(self) -> UniqueResult:
        """The first and last target indices for each source index.

        Returns:
            NamedTuple:
            - first: The first target index
            - last: The last target index
        """
        if self.src_indices is None:
            return UniqueResult(first=self.tgt_indices[:, 0], last=self.tgt_indices[:, -1])

        arg, diff0, diff1 = self._fl_mask
        tgt = self.tgt_indices[arg]
        return UniqueResult(first=tgt[diff0], last=tgt[diff1])

    @cached_property
    def local_index(self) -> LocalIndexResult:
        """The local indices of the source indices for each target index.

        Returns:
            NamedTuple:
            - floc: Local index in the first target
            - lloc: Local index in the last target
        """
        if self.src_indices is None:
            count = self.tgt_indices.shape[0]
            width = self.tgt_indices.shape[1]
            floc = bm.zeros((count,), dtype=bm.int32, device=self.tgt_indices.device)
            lloc = bm.full((count,), width - 1, dtype=bm.int32, device=self.tgt_indices.device)
            return LocalIndexResult(floc=floc, lloc=lloc)

        order = bm.argsort(self.tgt_indices)
        tgt = self.tgt_indices[order]
        idx = bm.arange(tgt.shape[0], dtype=bm.int32, device=tgt.device)
        TRUE = bm.ones((1,), dtype=bm.bool, device=tgt.device)
        diff = bm.concat([TRUE, tgt[1:] != tgt[:-1]])
        group = bm.cumsum(bm.astype(diff, bm.int32), axis=0) - 1
        group_start = idx[diff]
        loc = idx - group_start[group]
        loc = loc[bm.argsort(order)]

        arg, diff0, diff1 = self._fl_mask
        loc = loc[arg]

        return LocalIndexResult(floc=loc[diff0], lloc=loc[diff1])
