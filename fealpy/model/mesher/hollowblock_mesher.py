from typing import Optional, List, Tuple, Union

from ...backend import TensorLike, backend_manager as bm
from ...decorator import variantmethod
from ...mesh import TetrahedronMesh

import gmsh
import ast

class HollowBlockMesher:
    """
    Constructs a hollow rectangular block with cylindrical holes and provides access to mesh and hole boundary data.

    Parameters
        options : dict, optional
            Configuration for geometry and mesh generation. The dictionary can include:

            block : dict
                Outer block dimensions with keys:
                    - 'length' (float): Length in x-direction.
                    - 'width'  (float): Width in y-direction.
                    - 'height' (float): Height in z-direction.

            thickness : dict
                Wall thickness along each axis with keys:
                    - 'hx' (float): Inner wall offset in x-direction.
                    - 'hy' (float): Inner wall offset in y-direction.
                    - 'hz' (float): Inner wall offset in z-direction.

            cylinders : list of tuple
                Each item is ((x, y, z), r), defining a cylinder center and radius.

            h : float
                Maximum mesh element size.

            return_mesh : bool
                If True, generate and store the tetrahedral mesh.

            return_hole : bool
                If True, extract surface nodes on the cylindrical holes.

            show_figure : bool
                If True, show the geometry and mesh in Gmsh GUI.

    Attributes
        mesh : TetrahedronMesh or None
            The tetrahedral volume mesh, available only if return_mesh=True.
        holes : List[TensorLike] or None
            A list of node coordinates on the annular surfaces of cylindrical holes, available only if return_hole=True.
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

        self.options['cylinders'] = parse_opt(self.options.get('cylinders'))
        self.options['block'] = parse_opt(self.options.get('block'))
        self.options['thickness'] = parse_opt(self.options.get('thickness'))

        self.mesh: Optional[TetrahedronMesh] = None
        self.holes: Optional[List[TensorLike]] = None
        self.generate()
    
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the model.
        
        Returns
            dim : int
                The geometric dimension of the domain. Always returns 3.
        """
        3

    def get_holes(self, flags: Union[int, Tuple[int]]) -> Union[TensorLike, Tuple[TensorLike]]:
        """Return hole surface node coordinates for specified hole indices.

        Parameters
            flags : int or tuple of int
                The index or indices of the holes to retrieve.

        Returns
            holes : TensorLike or tuple of TensorLike
                The node coordinates on the annular surfaces of the specified holes.
        """
        option = self.options
        return_hole = option['return_hole']
        if not return_hole:
            raise RuntimeError("Please ensure that 'return_hole' is set to True.")
        
        if isinstance(flags, int):
            return self.holes[flags]

        return tuple(self.holes[i] for i in flags)

    def get_coupling_lines(self, flags: Union[int, Tuple[int]]) -> Union[TensorLike, Tuple[TensorLike]]:
        """Return REP2 lines: center to each surface node on hole(s).

        Parameters
            flags : int or tuple of int
                Index or indices of the holes.

        Returns
            lines : (N, 2, 3) array or tuple of such arrays
                Each line connects a center point to a surface node.
        """

        holes = self.get_holes(flags)
        block = self.options['block']
        thickness = self.options['thickness']
        cylinders = self.options['cylinders']
        width = block['width']
        hy = thickness['hy']
        
        if isinstance(flags, int):
            h = 0.5*hy if (flags % 2 == 0) else width - 0.5*hy
            center = bm.array(cylinders[flags//2][0], dtype=bm.float64) + bm.array([0, h, 0], dtype=bm.float64)
            lines = bm.concatenate([
                bm.broadcast_to(center, (len(holes), 1, 3)),
                holes[:, None, :]], axis=1)
            
            return lines
        
        results = []
        for i in range(len(flags)):
            flag = flags[i]
            h = 0.5*hy if (flag % 2 == 0) else width - 0.5*hy
            center = bm.array(cylinders[flag//2][0], dtype=bm.float64) + bm.array([0, h, 0], dtype=bm.float64)
            lines = bm.concatenate([
                bm.broadcast_to(center, (len(holes[i]), 1, 3)),
                holes[i][:, None, :]], axis=1)
            results.append(lines)

        return tuple(results)

    def get_options(self) -> dict:
        options = {
            'block': {
                'length': 10.0,
                'width': 6.0,
                'height': 20.0,
            },
            'thickness': {
                'hx': 1.1,
                'hy': 1.0,
                'hz': 1.0,
            },
            'cylinders': [
                ((4.0, 0, 13), 1.0),
                ((6.0, 0, 5), 1.0)
            ],
            'h': 0.8,
            'return_mesh': True,
            'return_hole': True,
            'show_figure': True,
        }

        return options

    def generate(self) -> None:
        """Generate a hollow block with cylindrical holes and extract mesh and/or hole surface nodes.
        """
        option = self.options
        block = option['block']
        thickness = option['thickness']
        cylinders = option['cylinders']
        h = option['h']
        return_mesh = option['return_mesh']
        return_hole = option['return_hole']
        show_figure = option['show_figure']
        
        length, width, height = block['length'], block['width'], block['height']
        hx, hy, hz = thickness['hx'], thickness['hy'], thickness['hz']
        
        gmsh.initialize()
        gmsh.model.add("HollowBlock")

        main_block = gmsh.model.occ.addBox(
            0.0, 0.0, 0.0,
            length,
            width,
            height
        )

        sub_block = gmsh.model.occ.addBox(
            hx, hy, hz,
            length - 2 * hx,
            width  - 2 * hy,
            height - 2 * hz
        )

        hollow_block = gmsh.model.occ.cut([(3, main_block)], [(3, sub_block)])
        gmsh.model.occ.synchronize()

        cylinder_tags = []
        for center, r in cylinders:
            cx, cy, cz = center
            tag = gmsh.model.occ.addCylinder(cx, cy - 0.1, cz, 0, width * 1.1, 0, r)
            cylinder_tags.append((3, tag))

        gmsh.model.occ.synchronize()
        gmsh.model.occ.cut(hollow_block[0], cylinder_tags)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2*h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")

        if return_hole:
            holes = []
            for center, r in cylinders:
                cx, cy, cz = center
                tol = 1e-02
                ring_front = gmsh.model.getEntitiesInBoundingBox(
                    cx - r - tol, cy - tol, cz - r - tol,
                    cx + r + tol, cy + hy + tol, cz + r + tol, dim=2)
                ring_back  = gmsh.model.getEntitiesInBoundingBox(
                    cx - r - tol, cy + width - hy - tol, cz - r - tol,
                    cx + r + tol, cy + width - hy + hy + tol, cz + r + tol, dim=2)

                ring_surface_tags = [ring_front[0][1], ring_back[0][1]]  # 圆环带的面 tag
                
                for tag in ring_surface_tags:
                    _, _, ntags = gmsh.model.mesh.getElements(2, tag)
                    node_ids = set()
                    for group in ntags:
                        node_ids.update(group)

                    node_ids = sorted(node_ids)
                    coords = [gmsh.model.mesh.getNode(nid)[0] for nid in node_ids]
                    holes.append(bm.array(coords))
            self.holes = holes   
        
        if return_mesh:
            ntags, vxyz, _ = gmsh.model.mesh.getNodes()
            node = vxyz.reshape((-1,3))
            vmap = dict({j:i for i,j in enumerate(ntags)})
            tets_tags,evtags = gmsh.model.mesh.getElementsByType(4)
            evid = bm.array([vmap[j] for j in evtags])
            cell = evid.reshape((tets_tags.shape[-1],-1))
            self.mesh = TetrahedronMesh(node, cell)

        if show_figure:
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.fltk.run()
        
        gmsh.finalize()

