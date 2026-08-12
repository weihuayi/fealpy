from .base import _MeshFactoryNewMixin


class PrismMesh(_MeshFactoryNewMixin):
    schema = "prism"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        threshold=None,
        device=None,
    ):
        """Create a prism mesh of the box."""
        from ...mesher.box import Box3d

        box = Box3d(box, nx, ny, nz, device=device)
        return box.prismatize()