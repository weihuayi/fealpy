import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        This script generates a mesh for a metalens using the MetalensesMesher class.
        """)
parser.add_argument('--backend',
        default='numpy', type=str,
        help='Default backend is numpy')
parser.add_argument('--glass_size',
        default=800, type=float,
        help='Size of the glass layer.')
parser.add_argument('--glass_height',
        default=3000, type=float,
        help='Height of the glass layer.')
parser.add_argument('--air_layer_height',
        default=4800, type=float,
        help='Height of the air layer.')
parser.add_argument('--bottom_pml_height',
        default=960, type=float,
        help='Height of the bottom PML layer.')
parser.add_argument('--top_pml_height',
        default=960, type=float,
        help='Height of the top PML layer.')
parser.add_argument('--antenna1_size',
        default=190, type=float,
        help='Size of the first antenna.')
parser.add_argument('--antenna1_height',
        default=600, type=float,
        help='Height of the first antenna.')
parser.add_argument('--antenna2_size',
        default=160, type=float,
        help='Size of the second antenna.')
parser.add_argument('--antenna2_height',
        default=600, type=float,
        help='Height of the second antenna.')
parser.add_argument('--antenna3_size',
        default=160, type=float,
        help='Size of the third antenna.')
parser.add_argument('--antenna3_height',
        default=600, type=float,
        help='Height of the third antenna.')
parser.add_argument('--antenna4_size',
        default=160, type=float,
        help='Size of the fourth antenna.')
parser.add_argument('--antenna4_height',
        default=600, type=float,
        help='Height of the fourth antenna.')
parser.add_argument('--mesh_type',
        default='tet', type=str,
        help='Type of mesh to generate, options are "tet" or "pri".')
parser.add_argument('--mesh_size',
        default=0.2, type=float,
        help='Size of the mesh elements.')
parser.add_argument('--translation_axes',
                    default=[1, 1, 0], type=tuple[int],
        help='Axes along which to translate the mesh, e.g., [1, 1, 0] for x, y, z.')
parser.add_argument('--translation_num',
                    default=[10, 10, 0], type=tuple[int],
        help='Number of translations along each axis, e.g., [10, 10, 0] for x, y, z.')
parser.add_argument('--output_to_vtu',
                    default=False, type=bool,
        help='Whether to output the mesh in VTU format.')

options = vars(parser.parse_args())


from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from fealpy.cem.mesh import MetalensesMesher

bm.set_backend(options['backend'])

metalenses_params = {
            "glass_size": options['glass_size'],
            "glass_height": options['glass_height'],
            "air_layer_height": options['air_layer_height'],
            "bottom_pml_height": options['bottom_pml_height'],
            "top_pml_height": options['top_pml_height'],
            "antenna1_size": options['antenna1_size'],
            "antenna1_height": options['antenna1_height'],
            "antenna2_size": options['antenna2_size'],
            "antenna2_height": options['antenna2_height'],
            "antenna3_size": options['antenna3_size'],
            "antenna3_height": options['antenna3_height'],
            "antenna4_size": options['antenna4_size'],
            "antenna4_height": options['antenna4_height']}

metalenses_mesher = MetalensesMesher(metalenses_params, options['mesh_type'])

unit_tet_mesh = metalenses_mesher.generate(options['mesh_size'])
node_pairs = unit_tet_mesh.meshdata['node_pairs']

total_mesh = metalenses_mesher.assemble_total_mesh(unit_tet_mesh,
                                                   options['mesh_type'],
                                                   node_pairs,
                                                   options['translation_axes'],
                                                   options['translation_num'])
if options['output_to_vtu']:
    unit_tet_mesh.to_vtk(fname='unit_tet_mesh.vtu')
    total_mesh.to_vtk(fname='total_mesh.vtu')
