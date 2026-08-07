import numpy as np

from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh


class CylinderMesher:
    """First-order tetrahedral mesh generator for a straight cylinder."""

    def __init__(self, radius=None, height=None, lc=None):
        self.radius = 1.0 if radius is None else float(radius)
        self.height = 1.0 if height is None else float(height)
        self.lc = 0.3 if lc is None else float(lc)
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.height <= 0.0:
            raise ValueError("height must be positive.")
        if self.lc <= 0.0:
            raise ValueError("lc must be positive.")

    def geo_dimension(self) -> int:
        return 3

    @variantmethod("tet")
    def init_mesh(self):
        """Generate a current-schema tetrahedral mesh through Gmsh."""
        import gmsh

        was_initialized = bool(gmsh.isInitialized())
        if not was_initialized:
            gmsh.initialize()
        previous_terminal = gmsh.option.getNumber("General.Terminal")
        previous_order = gmsh.option.getNumber("Mesh.ElementOrder")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.clear()
        gmsh.model.add("Cylinder")
        try:
            gmsh.model.occ.addCylinder(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                self.height,
                self.radius,
            )
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.lc)
            gmsh.model.mesh.generate(3)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            tag_to_index = {
                int(tag): index for index, tag in enumerate(node_tags)
            }
            _, connectivity = gmsh.model.mesh.getElementsByType(4)
            if connectivity.size == 0:
                raise RuntimeError("Gmsh generated no first-order tetrahedra.")
            gmsh_cells = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
            cell = np.fromiter(
                (
                    tag_to_index[int(tag)]
                    for element in gmsh_cells
                    for tag in element
                ),
                dtype=np.int32,
                count=gmsh_cells.size,
            ).reshape(-1, 4)
            node = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            return TetrahedronMesh(
                bm.array(node, dtype=bm.float64),
                bm.array(cell, dtype=bm.int32),
            )
        finally:
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", previous_terminal)
            gmsh.option.setNumber("Mesh.ElementOrder", previous_order)
            if not was_initialized:
                gmsh.finalize()
