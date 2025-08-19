from typing import Optional, List, Tuple, Union

from ..backend import TensorLike, backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh
from ..geometry import DLDModeler

import gmsh
import ast


class DLDMesher:
    """
    A class to generate a microfluidic chip mesh with periodic circular holes.

    This class initializes a 2D rectangular domain with a grid of circular holes arranged in a staggered pattern. 
    The geometry is defined using Gmsh, and a triangular mesh is generated over the remaining area.

    Parameters:
        options (dict, optional): Configuration for geometry and meshing. If not provided, default parameters are used.

    Attributes:
        box (List[float]): The bounding rectangle of the chip domain [x0, x1, y0, y1].
        center (Tuple[float, float]): Starting point (x, y) for the first circle.
        r (float): Radius of the circles.
        l1 (float): Vertical distance between adjacent circles in the same column.
        l2 (float): Horizontal distance between adjacent columns.
        h (float): Vertical offset applied to each subsequent column.
        lc (float): Target mesh size for Gmsh meshing.
        centers (List[Tuple[float, float]]): Computed centers of all valid circles inside the box.
        mesh (TriangleMesh): The final mesh generated (if `return_mesh=True`).
    """
    def __init__(self, options: Optional[dict] = None):
        self.options = options
        if not options:
            self.options = self.get_options()

        self.options['return_project_edge'] = self._parse_opt(self.options.get('return_project_edge'))
        self.options['return_mesh'] = self._parse_opt(self.options.get('return_mesh'))
        self.options['show_figure'] = self._parse_opt(self.options.get('show_figure'))
    
    def _parse_opt(self, opt):
        """If the input is a string, try to convert it to a Python object.

        Parameters
            opt : Any
                Input value, possibly a string representing a Python literal.

        Returns
            Any
                Parsed Python object or the original value.
        """
        if isinstance(opt, str):
            try:
                return ast.literal_eval(opt)
            except Exception:
                return opt
        return opt
    
    def get_options(self) -> dict:
        options = {
            'lc': 0.03,
            'hole_lc': 0.01,
            'return_mesh': True,
            'show_figure': False,
        }

        return options
    
    @variantmethod('tri')
    def generate(self, modeler: DLDModeler, gmsh):
        import gmsh
        option = self.options
        r = option['radius']
        lc = option['lc']
        hole_lc = option['hole_lc']
        return_project_edge = option.get('return_project_edge', None)
        refine = option.get('refine', None)
        return_mesh = option['return_mesh']
        show_figure = option['show_figure']

        centers = modeler.centers

        if refine:
            f_dist = gmsh.model.mesh.field.add("Distance")
            circle_edges = []
            for cx, cy in centers:
                ctag = gmsh.model.occ.addCircle(cx, cy, 0, r)
                circle_edges.append(ctag)

            gmsh.model.occ.synchronize()
            gmsh.model.mesh.field.setNumbers(f_dist, "EdgesList", circle_edges)

            f_th = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_th, "SizeMin", hole_lc) 
            gmsh.model.mesh.field.setNumber(f_th, "SizeMax", lc)
            gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0)
            gmsh.model.mesh.field.setNumber(f_th, "DistMax", r * 1.5)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

        else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()


        if return_project_edge:   
            holes = []
            for center in centers:
                cx, cy = center
                tol = lc / 10 # 确保与 lc 同阶
                 
                mark = gmsh.model.getEntitiesInBoundingBox(
                    cx - r - tol, cy - r - tol, -1, 
                    cx + r + tol, cy + r + tol, 1, dim=1)

                tag = mark[0][1]  # 圆 tag
                

                _, _, ntags = gmsh.model.mesh.getElements(1, tag)
                node_ids = set()
                for group in ntags:
                    node_ids.update(group)

                node_ids = sorted(node_ids)
                coords = [gmsh.model.mesh.getNode(nid)[0] for nid in node_ids]
                holes.append(bm.array(coords)[:, :2])
        
        if return_mesh:
            
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = bm.from_numpy(node_coords.reshape((-1, 3))[:, :2])
            nodetags_map = dict({j: i for i, j in enumerate(node_tags)})
            cell_type = 2
            cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

            evid = bm.array([nodetags_map[j] for j in cell_connectivity])
            cell = evid.reshape((cell_tags.shape[-1], -1))
            
            unique, inverse = bm.unique(cell.ravel(), return_inverse=True)
            new_cell = inverse.reshape(cell.shape)

            self.mesh = TriangleMesh(node[unique], new_cell)

        if show_figure:
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.fltk.run()
        
    @generate.register('quad')
    def generate(self):
        pass