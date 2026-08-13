from pathlib import Path

from ...backend import Tensor
from ..storage import EntitySector, MeshBlock
from ..topology.builder import TopologyBuilder
from ..view import Mesh


class MeshFactory(type):
    schema: str

    def __instancecheck__(cls, instance) -> bool:
        if not isinstance(instance, Mesh):
            return False
        return instance.is_elemental(cls.schema)


class _MeshFactoryNewMixin(metaclass=MeshFactory):
    def __new__(cls, node: Tensor, cell: Tensor) -> Mesh:
        block = MeshBlock(positions=node)
        block.add_sector(EntitySector(cls.schema, cell), root=True)
        TopologyBuilder.construct(block)
        return Mesh(block)

    @classmethod
    def read(cls, filename: str | Path, file_format: str | None = None) -> Mesh:
        """Read a mesh from a file and return a new Mesh instance."""
        from ..mesh_io import read

        block = read(filename, file_format=file_format)
        return Mesh(block)

    @classmethod
    def write(
        cls,
        filename: str | Path,
        mesh: Mesh,
        file_format: str | None = None,
        **kwargs,
    ) -> None:
        """Write a mesh to a file."""
        from ..mesh_io import write

        if not isinstance(mesh, Mesh):
            raise TypeError(f"Expected a Mesh instance, got {type(mesh)}")
        if not mesh.is_elemental(cls.schema):
            raise ValueError(f"Mesh is not of type {cls.schema}")
        write(filename, mesh.block, [cls.schema], file_format=file_format, **kwargs)