from typing import Optional, List, Tuple, Union

from ..backend import TensorLike, backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh

import gmsh
import ast

class ChipMesher:
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

        def parse_opt(opt):
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

        self.options['return_mesh'] = parse_opt(self.options.get('return_mesh'))
        self.options['show_figure'] = parse_opt(self.options.get('show_figure'))

        self.box = self.options['box']
        self.generate()
    
    def get_options(self) -> dict:
        options = {
            'box': [0.0, 0.75, 0.0, 0.41],
            'center': (0.1, 0.05),
            'radius': 0.029,
            'l1': 0.12,
            'l2': 0.12,
            'h': 0.04,
            'lc': 0.03,
            'hole_lc': 0.01,
            'return_mesh': True,
            'show_figure': False,
        }

        return options
    
    def generate(self):
        option = self.options
        self.box = option['box']
        self.center = option['center']
        self.r = option['radius']
        self.l1 = option['l1']
        self.l2 = option['l2']
        self.h = option['h']
        self.lc = option['lc'] 
        self.hole_lc = option['hole_lc'] 

        return_mesh = option['return_mesh']
        show_figure = option['show_figure']

        gmsh.initialize()
        gmsh.model.add("chip")
        
        # 定义矩形区域
        x0, x1, y0, y1 = self.box
        rectangle = gmsh.model.occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)

        # 生成圆形孔洞的中心位置
        self.centers = self.generate_circle_centers(self.box, self.center, self.l1, self.l2, self.h)
        circle_tags = []
        for cx, cy in self.centers:
            circle = gmsh.model.occ.addDisk(cx, cy, 0, self.r, self.r)
            circle_tags.append(circle)

        gmsh.model.occ.cut([(2, rectangle)], [(2, tag) for tag in circle_tags], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # 定义距离场，用于孔洞附近网格加密
        f_dist = gmsh.model.mesh.field.add("Distance")
        circle_edges = []
        for cx, cy in self.centers:
            ctag = gmsh.model.occ.addCircle(cx, cy, 0, self.r)
            circle_edges.append(ctag)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.setNumbers(f_dist, "EdgesList", circle_edges)

        # 定义阈值场，专门调整孔洞附近的网格
        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", self.hole_lc)  # 孔洞附近的小网格
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", self.lc)  # 其他地方的网格
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0)   # 孔洞边界内部
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", self.r * 1.5)  # 孔洞半径的1.5倍内细网

        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

        # 生成网格
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()

        if return_mesh:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = node_coords.reshape((-1, 3))[:, :2]
            nodetags_map = dict({j: i for i, j in enumerate(node_tags)})
            cell_type = 2
            cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

            evid = bm.array([nodetags_map[j] for j in cell_connectivity])
            cell = evid.reshape((cell_tags.shape[-1], -1))

            unique, inverse = bm.unique(cell, return_inverse=True)
            new_cell = inverse.reshape(cell.shape)

            self.mesh = TriangleMesh(node[unique], new_cell)

        if show_figure:
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.fltk.run()

        gmsh.finalize()

    def generate_circle_centers(self, box, center, l1, l2, h):
        x0, x1, y0, y1 = box
        cx0, cy0 = center
        r = self.r

        centers = []

        x_min = x0 - 2 * r
        x_max = x1 + 2 * r
        y_min = y0 - 2 * r
        y_max = y1 + 2 * r

        col_idx = 0
        x = cx0
        while x <= x_max:
            y_start = cy0 + col_idx * h
            y_up = y_start
            y_down = y_start - l1
            while y_up <= y_max:
                if not (x + r < x0 or x - r > x1 or y_up + r < y0 or y_up - r > y1):
                    centers.append((x, y_up))
                y_up += l1
            while y_down >= y_min:
                if not (x + r < x0 or x - r > x1 or y_down + r < y0 or y_down - r > y1):
                    centers.append((x, y_down))
                y_down -= l1
            x += l2
            col_idx += 1
        return centers
