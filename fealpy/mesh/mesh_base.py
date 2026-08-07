from .view import Mesh

__all__ = [
    "Mesh",
    "HomogeneousMesh",
    "SimplexMesh",
    "TensorMesh",
    "StructuredMesh",
]

# deprecated, will be removed in future versions
class _HomogeneousMeshMeta(type):
    """Homogeneous mesh."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh)

HomogeneousMesh = _HomogeneousMeshMeta("HomogeneousMesh", (Mesh,), {})


class _SimplexMeshMeta(type):
    """Simplex mesh."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh) and instance.is_simplex_mesh()

SimplexMesh = _SimplexMeshMeta("SimplexMesh", (Mesh,), {})


class _TensorMeshMeta(type):
    """Tensor mesh."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh) and instance.is_tensor_mesh()

TensorMesh = _TensorMeshMeta("TensorMesh", (Mesh,), {})


class _StructuredMeshMeta(type):
    """Structured mesh."""
    def __instancecheck__(self, instance):
        return isinstance(instance, Mesh)
    # TODO: change after we have structured mesh implementation

StructuredMesh = _StructuredMeshMeta("StructuredMesh", (Mesh,), {})
