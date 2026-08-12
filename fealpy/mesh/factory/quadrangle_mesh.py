from .base import _MeshFactoryNewMixin


class QuadrangleMesh(_MeshFactoryNewMixin):
    schema = "quad"

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
        """Create a quadrangle mesh of the box."""
        from ...mesher.box import Box2d

        box = Box2d(box, nx, ny, device=device)
        return box.quadrangulate()