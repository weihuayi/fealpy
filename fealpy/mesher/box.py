
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ..backend import bm, Tensor
from ..mesh import Mesh, MeshBlock, EntitySector


class BoxCache(NamedTuple):
    node: Tensor
    cell: Tensor


@dataclass(slots=True)
class Box1d:
    box: list[float] = field(default_factory=list)
    nx: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1]

    def initialize(self):
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx

        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        node = bm.reshape(x, (-1, 1))
        idx = bm.arange(nx + 1, dtype=bm.int32, device=self.device)
        cell = bm.stack((idx[:-1], idx[1:]), axis=1)

        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        self._cache = None

    def nodalize(self) -> Mesh:
        """Create a mesh of the box with positions only."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return Mesh(block).construct()

    def segmentize(self) -> Mesh:
        """Create a segmented mesh of the box."""
        node, cell = self.initialize()

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("segment", cell), root=True)
        return Mesh(block).construct()


@dataclass(slots=True)
class Box2d:
    box: list[float] = field(default_factory=list)
    nx: int = 10
    ny: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1, 0, 1]

    def initialize(self):
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx
        ny = self.ny

        NN = (nx + 1) * (ny + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64, device=self.device)
        X, Y = bm.meshgrid(x, y, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32, device=self.device), (nx + 1, ny + 1))

        cell0 = idx[:-1, :-1] # type: ignore
        cell1 = cell0 + ny + 1
        cell2 = cell1 + 1
        cell3 = cell0 + 1
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
            ),
            axis=1,
        )
        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        self._cache = None

    def nodalize(self) -> Mesh:
        """Create a mesh of the box with positions only."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return Mesh(block).construct()

    def triangulate(self) -> Mesh:
        """Create a triangulated mesh of the box."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 3],
            [1, 2, 3],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 3)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("tri", cell), root=True)
        return Mesh(block).construct()

    def quadrangulate(self) -> Mesh:
        """Create a quadrilateral mesh of the box."""
        node, cell = self.initialize()
        cell = bm.reshape(cell, (-1, 4))

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("quad", cell), root=True)
        return Mesh(block).construct()


@dataclass(slots=True)
class Box3d:
    box: list[float] = field(default_factory=list)
    nx: int = 10
    ny: int = 10
    nz: int = 10
    device: Any | None = None
    _cache: BoxCache | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.box:
            self.box = [0, 1, 0, 1, 0, 1]

    def initialize(self):
        if self._cache is not None:
            return self._cache.node, self._cache.cell

        box = self.box
        nx = self.nx
        ny = self.ny
        nz = self.nz

        NN = (nx + 1) * (ny + 1) * (nz + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=self.device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64, device=self.device)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=bm.float64, device=self.device)
        X, Y, Z = bm.meshgrid(x, y, z, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
                bm.reshape(Z, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32, device=self.device), (nx + 1, ny + 1, nz + 1))

        nyz = (ny + 1) * (nz + 1)
        cell0 = idx[:-1, :-1, :-1] # type: ignore
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
                bm.reshape(cell4, (-1, 1)),
                bm.reshape(cell5, (-1, 1)),
                bm.reshape(cell6, (-1, 1)),
                bm.reshape(cell7, (-1, 1)),
            ),
            axis=1,
        )
        self._cache = BoxCache(node=node, cell=cell)
        return node, cell

    def clear(self) -> None:
        self._cache = None

    def nodalize(self) -> Mesh:
        """Create a mesh of the box with positions only."""
        node, _ = self.initialize()

        block = MeshBlock(positions=node)
        return Mesh(block).construct()

    def tetrahedralize(self) -> Mesh:
        """Create a tetrahedral mesh of the box."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 3, 4],
            [1, 2, 3, 5],
            [4, 7, 5, 3],
            [6, 5, 7, 2],
            [1, 5, 3, 4],
            [3, 2, 7, 5]
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 4)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("tet", cell), root=True)
        return Mesh(block).construct()

    def prismatize(self) -> Mesh:
        """Create a prismatic mesh of the box."""
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 4, 5, 6],
            [0, 2, 3, 4, 6, 7]
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 6)) # type: ignore

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("prism", cell), root=True)
        return Mesh(block).construct()

    def hexahedralize(self) -> Mesh:
        """Create a hexahedral mesh of the box."""
        node, cell = self.initialize()
        cell = bm.reshape(cell, (-1, 8))

        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("hex", cell), root=True)
        return Mesh(block).construct()
