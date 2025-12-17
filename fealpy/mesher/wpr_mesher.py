from typing import Any, Dict, Optional, List, Tuple, Union

import ast

from ..backend import TensorLike, backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh
import gmsh


class WPRMesher:
    def __init__(self, options: Optional[dict] = None):
        """
        Initialize the Water Purification Reactor mesher.
        
        Parameters:
            options: Configuration dictionary for meshing parameters.
        """
        if not options:
            self.options = self.get_options()
        else:
            self.options = options

        self.options['return_mesh'] = self.options.get('return_mesh', True)
        self.options['show_figure'] = self.options.get('show_figure', False)

        self.mesh: Optional[TriangleMesh] = None

    def get_options(self) -> dict:
        options = {
            'block_length': 6.0,
            'block_width': 2.0,
            'inlet_length': 0.5,
            'inlet_width': 0.5,
            'gap': 0.1,
            'gap_len': 1,
            'h': 0.6,
            'return_mesh': True,
            'show_figure': False,
        }

        return options

    def generate(self) -> None:
        option = self.options
        block_length = option['block_length']
        block_width = option['block_width']
        inlet_length = option['inlet_length']
        inlet_width = option['inlet_width']

        gap = option['gap']
        lc = option['lc']
        
        return_mesh = option['return_mesh']
        show_figure = option['show_figure']
        
        gmsh.initialize()
        gmsh.model.add("WPR")

        main_block = gmsh.model.occ.addRectangle(
            0.0, 0.0, 0.0,
            block_length,
            block_width
        )

        sub_block0 = gmsh.model.occ.addRectangle(
            0, 0, 0,
            inlet_length,
            1 - 0.5 * inlet_width
        )

        sub_block1 = gmsh.model.occ.addRectangle(
            0, 1 + 0.5 * inlet_width, 0,
            inlet_length,
            1 - 0.5 * inlet_width
        )

        sub_block2 = gmsh.model.occ.addRectangle(
            5.5, 0, 0,
            inlet_length,
            1 - 0.5 * inlet_width
        )

        sub_block3 = gmsh.model.occ.addRectangle(
            5.5, 1 + 0.5 * inlet_width, 0,
            inlet_length,
            1 - 0.5 * inlet_width
        )

        # block = gmsh.model.occ.cut([(2, main_block)], [(2, sub_block0), (2, sub_block1), (2, sub_block2), (2, sub_block3)])
        block = main_block
        gmsh.model.occ.synchronize()
        len = option['gap_len']
        slit0 = gmsh.model.occ.addRectangle(
            1.5, 2 - len, 0,
            gap,
            len
        )

        slit1 = gmsh.model.occ.addRectangle(
            2.5, 0, 0,
            gap,
            len
        )

        slit2 = gmsh.model.occ.addRectangle(
            3.5, 2 - len, 0,
            gap,
            len
        )

        slit3 = gmsh.model.occ.addRectangle(
            4.5, 0, 0,
            gap,
            len
        )

        block = gmsh.model.occ.cut([(2, main_block)], 
                                   [
                                    (2, slit0), (2, slit1),
                                    (2, slit2), (2, slit3),
                                    (2, sub_block0), (2, sub_block1), 
                                    (2, sub_block2), (2, sub_block3)
                                    ])
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2*lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        
        if return_mesh:
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node = node_coords.reshape((-1,3))[:,:2]
            nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

            cell_type = 2
            cell_tags,cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

            evid = bm.array([nodetags_map[j] for j in cell_connectivity])
            cell = evid.reshape((cell_tags.shape[-1],-1))
            self.mesh = TriangleMesh(node,cell)

        if show_figure:
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.fltk.run()
        
        gmsh.finalize()




