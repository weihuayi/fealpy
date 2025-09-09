from typing import Any, Dict, Optional, List, Tuple, Union

import ast

from ..backend import TensorLike, backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh
from ..geometry import DLDMicrofluidicChipModeler


class DLDMicrofluidicChipMesher:
    """
    A mesher class for generating finite element meshes for DLD microfluidic chips.

    Parameters:
        options (Optional[Dict[str, Any]]): Configuration dictionary for meshing
            parameters. If None, default parameters are used.

    Attributes:
        options (Dict[str, Any]): Meshing configuration parameters.
        mesh (Optional[TriangleMesh]): The generated triangular mesh.
        project_edges (Optional[List[TensorLike]]): List of edge arrays for pillar boundaries.
    
    Examples:
        >>> options = {}
        >>> gmsh.initialize()
        >>> modeler = DLDMicrofluidicChipModeler(options)
        >>> modeler.build(gmsh_instance)
        >>> mesher = DLDMicrofluidicChipMesher(options)
        >>> mesher.generate(modeler, gmsh_instance)
        >>> mesh = mesher.mesh
        >>> gmsh.finalize()
    """
    
    def __init__(self, options: Optional[dict] = None):
        """
        Initialize the DLD microfluidic chip mesher.
        
        Parameters:
            options: Configuration dictionary for meshing parameters.
        """
        if not options:
            self.options = self.get_options()
        else:
            self.options = options

        self.options['return_mesh'] = self._parse_opt(self.options.get('return_mesh', True))
        self.options['show_figure'] = self._parse_opt(self.options.get('show_figure', False))

        self.mesh: Optional[TriangleMesh] = None
        self.project_edges: Optional[List[TensorLike]] = None

    def _parse_opt(self, opt: Any) -> Any:
        """
        Parse option values, converting string representations to Python objects.
        
        Parameters:
            opt: Input value, possibly a string representing a Python literal.
            
        Returns:
            Parsed Python object or the original value.
        """
        if isinstance(opt, str):
            try:
                return ast.literal_eval(opt)
            except Exception:
                return opt
        return opt
    
    def get_options(self) -> Dict[str, Any]:
        """
        Return the default option dictionary for meshing configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing default meshing parameters:
            
            - radius (float): Radius of the pillars (default: 0.1)
            - lc (float): Target characteristic mesh size (default: 0.3)
            - local_refine (bool): Enable local refinement around pillars (default: False)
            - return_project_edges (bool): Return pillar boundary edges (default: True)
            - return_mesh (bool): Return TriangleMesh object (default: True)
            - show_figure (bool): Display mesh figure using Gmsh GUI (default: False)
        """
        return {
            'radius': 0.1,
            'lc': 0.3,
            'local_refine': False,
            'return_project_edges': True,
            'return_mesh': True,
            'show_figure': False,
        }

    def get_node_data(self, gmsh: Any = None) -> Tuple[TensorLike, Dict[int, int]]:
        """
        Extracts node coordinates and creates a mapping of node tags to indices.

        Parameters:
            gmsh: The GMSH model instance.

        Returns:
            node: A tensor of node coordinates (x, y).
            nodetags_map: A dictionary mapping node tags to indices.
        """
        try:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            if node_tags.size == 0:
                raise RuntimeError("No nodes found in Gmsh model")
                
            node = bm.from_numpy(node_coords.reshape((-1, 3))[:, :2])
            nodetags_map = {tag: idx for idx, tag in enumerate(node_tags)}
            
            return node, nodetags_map
        except Exception as e:
            raise RuntimeError(f"Failed to extract node data from Gmsh: {e}")

    def generate(self, modeler: DLDMicrofluidicChipModeler, gmsh: Any = None) -> None:
        """
        Generate the mesh for the DLD microfluidic chip.
        
        Parameters:
            modeler: DLDMicrofluidicChipModeler instance with built geometry.
            gmsh: GMSH instance for mesh generation.
        """
        if gmsh is None:
            raise TypeError("A valid GMSH instance must be provided for mesh generation.")
        if modeler.circles is None:
            raise ValueError("Modeler has not been built. Call modeler.build() first.")
        
        option = self.options
        radius: float = option['radius']
        lc: float = option['lc']
        local_refine: bool = option.get('local_refine', False)
        return_mesh: bool = option.get('return_mesh', True)
        show_figure: bool = option.get('show_figure', False)
        return_project_edges: bool = option.get('return_project_edges', True)

        centers: TensorLike = modeler.circles[:, :2]  # Extract (x, y) coordinates
        self.radius = radius
        self.centers = centers
        self.boundary = modeler.boundary
        self.inlet_boundary = modeler.inlet_boundary
        self.outlet_boundary = modeler.outlet_boundary
        self.wall_boundary = modeler.wall_boundary
        try:
            if local_refine:
                self.local_refine(gmsh, centers, radius, lc)
            else:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
           
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.model.mesh.generate(2)
            gmsh.model.occ.synchronize()

            if return_project_edges:
                    self.project_edges = self.get_project_edges(
                        gmsh, centers, radius, lc
                    )
            
            if return_mesh:
                    self.mesh = self.build_mesh(gmsh)
            
            if show_figure:
                    self.show(gmsh)

        except Exception as e:
            raise RuntimeError(f"Mesh generation failed: {e}")

    def local_refine(self, gmsh: Any, centers: TensorLike, radius: float, lc: float) -> None:
        """Set up local mesh refinement around pillars using Gmsh distance fields."""

        f_dist = gmsh.model.mesh.field.add("Distance")
        circle_edges = []

        for cx, cy in centers:
            ctag = gmsh.model.occ.addCircle(cx, cy, 0, radius)
            circle_edges.append(ctag)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.setNumbers(f_dist, "EdgesList", circle_edges)

        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", lc / 3) 
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", lc)
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", radius * 1.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

    def get_project_edges(self, gmsh: Any, centers: TensorLike, radius: float, lc: float) -> TensorLike:
        """Extract boundary edges for all pillars."""

        project_edges = []
        node, nodetags_map = self.get_node_data(gmsh)
        tol: float = lc / 10

        for center in centers:
            cx, cy = center[:2]
            
            mark = gmsh.model.getEntitiesInBoundingBox(
                cx - radius - tol, cy - radius - tol, -1, 
                cx + radius + tol, cy + radius + tol, 1, 
                dim=1
            )

            if not mark:
                continue

            tag = mark[0][1]
            _, _, ntags = gmsh.model.mesh.getElements(1, tag)

            edges = bm.array([nodetags_map[j] for j in ntags[0]]).reshape(-1, 2)
            project_edges.append(edges)
        
        return bm.stack(project_edges, axis=0)

    def build_mesh(self, gmsh: Any) -> TriangleMesh:
        """Create TriangleMesh from Gmsh mesh data."""

        node, nodetags_map = self.get_node_data(gmsh)
        cell_type = 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
        
        if cell_tags.size == 0:
            raise RuntimeError("No triangular elements found in mesh")
        
        evid = bm.array([nodetags_map[j] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1], -1))
        
        unique, inverse = bm.unique(cell.ravel(), return_inverse=True)
        new_cell = inverse.reshape(cell.shape)
        
        return TriangleMesh(node[unique], new_cell)

    def show(self, gmsh: Any) -> None:
        """
        Display the mesh using Gmsh GUI.
        
        Parameters:
            gmsh: GMSH instance.
        """
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.option.setNumber("Mesh.VolumeEdges", 0)
        gmsh.fltk.run()

        