
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, overload, Self

from ...backend import Tensor
from ..schema import registry as _Reg
from ..storage import MeshBlock
from .entity_view import EntityView


class AdjointRelation(NamedTuple):
    src_indices: Tensor | None
    tgt_indices: Tensor


@dataclass
class MeshView:
    """Provides a view of the mesh topology and geometry."""
    block: MeshBlock

    ## Entity getters

    @overload
    def Entity(self, name_or_topdim: str | int, /) -> EntityView: ...
    @overload
    def Entity(self, name_or_topdim: str | int, idx: int, /) -> EntityView: ...
    def Entity(self, name_or_topdim: str | int, idx: int = 0, /) -> EntityView:
        """Get an entity view by its name or top dimension.

        Parameters:
            name_or_topdim (str | int): The name or top dimension of the entity.
                It can be:
                - the name of the entity (e.g., "point", "segment", "tri", "quad", "tet", "hex"),
                - the entity type with an optional index (e.g., "cell:0", "face:1", "edge:2", "node:3"),
                - just the entity type (e.g., "cell", "face", "edge", "node"),
                - the top dimension as an integer (e.g., 0 for nodes, 1 for edges).
                Where the negative top dimension counts from the end (e.g., -1 for cells, -2 for faces).
            idx (int, optional): The index to select a type of entity of the specified top dimension.
                The negative index counts from the end. Default is 0, which selects the first entity of that dimension.
                Ignored if `name_or_topdim` is the name or a string that specifies the index.

        Returns:
            EntityView: The requested entity view.

        Example:
            To get the QUAD from a Prism mesh, these are equivalent:
            >>> mesh.Entity("quad")
            >>> mesh.Entity("face:1") # while "face:0" would return the TRI
            >>> mesh.Entity("face", 1)
            >>> mesh.Entity("face", -1)
            >>> mesh.Entity(2, 1)
            >>> mesh.Entity(-2, 1)
            >>> mesh.Quad
            >>> mesh.Faces[1]
        """
        name = _Reg.schema_name_single_parser(
            name_or_topdim, idx, self.top_dimension(), self.block.sectors.keys()
        )
        return EntityView(self.block, self.block.sectors[name])

    def Entities(self, etype_or_topdim: str | int, /) -> list[EntityView]:
        """Get all entity views of a given type or top dimension.

        Parameters:
            etype_or_topdim (str | int): The entity type or top dimension.
                It can be:
                - the name of the entity (e.g., "point", "segment", "tri", "quad", "tet", "hex"),
                - the entity type with an optional index (e.g., "cell:0", "face:1", "edge:2", "node:3"),
                - just the entity type (e.g., "cell", "face", "edge", "node"),
                - the top dimension as an integer (e.g., 0 for nodes, 1 for edges).
                Where the negative top dimension counts from the end (e.g., -1 for cells, -2 for faces).

        Returns:
            list[EntityView]: A list of entity views corresponding to the specified type or top dimension.
        """
        if isinstance(etype_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(etype_or_topdim, self.top_dimension())
        else:
            if not isinstance(etype_or_topdim, str):
                raise TypeError(f"Expected str or int for etype_or_topdim, got {type(etype_or_topdim)}")
            if etype_or_topdim in self.block.sectors:
                return [EntityView(self.block, self.block.sectors[etype_or_topdim])]
            topdim = _Reg.etype_to_topdim(etype_or_topdim, self.top_dimension())

        names = _Reg.topdim_to_names(topdim, self.block.sectors.keys())
        return [EntityView(self.block, self.block.sectors[name]) for name in names]

    ## Checkers

    def is_simplex_mesh(self) -> bool:
        """Check if the mesh is a simplex mesh."""
        for name in self.block.sectors.keys():
            if name not in {"point", "segment", "tri", "tet"}:
                return False
        return True

    def is_tensor_mesh(self) -> bool:
        """Check if the mesh is a tensor mesh."""
        for name in self.block.sectors.keys():
            if name not in {"point", "segment", "quad", "hex"}:
                return False
        return True

    def is_elemental(self, entity_name: str | None = None, /) -> bool:
        """Check if the mesh has only one root entity type."""
        is_one_root = len(self.block.root_entity_names) == 1
        if entity_name is None:
            return is_one_root
        return is_one_root and self.block.root_entity_names[0] == entity_name

    ## Other getters

    def geo_dimension(self) -> int:
        """Get the geometric dimension of the mesh."""
        return int(self.block.positions.shape[1])

    def fealpy_api(self):
        """Provides a view of the mesh compatible with FEALPy's API."""
        from .fealpy_api import Mesh
        return Mesh(self.block)

    def top_dimension(self) -> int:
        """Get the topological dimension of the mesh."""
        if not self.block.root_entity_names:
            return -1
        return max(self.block.sectors[name].schema.top_dim
                   for name in self.block.root_entity_names)

    # Setters (in-place modification)

    def construct(self, exclude: list[str] | None = None) -> Self:
        """Construct the mesh topology, optionally excluding certain entity types."""
        from ..topology.builder import TopologyBuilder
        TopologyBuilder.construct(self.block, exclude=exclude)
        return self

    def uniform_refine(self, times: int = 1, **kwargs):
        """Uniformly refine the mesh a given number of times.

        Parameters:
            times: Number of times to refine the mesh. Default is 1.
            **kwargs: Additional arguments for the refinement process.
        """
        from ..uniform_refine import uniform_refine
        return uniform_refine(self.block, times=times, **kwargs)
