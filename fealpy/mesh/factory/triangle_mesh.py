from .base import _MeshFactoryNewMixin


class TriangleMesh(_MeshFactoryNewMixin):
    schema = "tri"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1],
        nx=10,
        ny=10,
        *,
        threshold=None,
        device=None,
    ):
        """Create a triangle mesh of the box."""
        from ...mesher.box import Box2d

        box = Box2d(box, nx, ny, device=device)
        return box.triangulate()