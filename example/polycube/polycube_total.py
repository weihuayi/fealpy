from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.polycube import PolyCubeProcessor

import pickle
import json


def str_to_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid dict string: {s}") from e

import argparse
# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        This script demonstrates the use of the PolyCubeProcessor class
        """)

parser.add_argument('--backend',
        default='pytorch', type=str,
        help='Default backend is pytorch')
parser.add_argument('--normal_smooth_sigma',
                    default=0.1, type=float,
        help='Sigma for Gaussian kernel in normal smooth deformation')
parser.add_argument('--normal_smooth_s',
                    default=7, type=int,
        help='Number of nearest neighbors for normal smooth deformation')
parser.add_argument('--normal_smooth_alpha',
                    default=0.5, type=float,
        help='Weight for the smoothness term in normal smooth deformation')
parser.add_argument('--normal_smooth_max_epochs',
                    default=100000, type=int,
        help='Maximum epochs for normal smooth deformation')
parser.add_argument('--normal_smooth_error_threshold',
                    default=1e-3, type=float,
        help='Error threshold for convergence in normal smooth deformation')
parser.add_argument('--normal_smooth_weights',
                    default='{"node": 1.0, "rotate_matrix": 0.0}', type=str_to_dict,
        help='Weights for different components in the deformation')
parser.add_argument('--straighten_max_iter',
                    default=5, type=int,
        help='Maximum iterations for polycube straightening')
parser.add_argument('--merge_min_size',
                    default=5, type=int,
        help='Minimum size for merging segments in polycube segmentation')
parser.add_argument('--laplacian_smooth_alpha',
                    default=0.3, type=float,
        help='Alpha for Laplacian smoothing in polycube segmentation')
parser.add_argument('--laplacian_smooth_max_iter',
                    default=100, type=int,
        help='Maximum iterations for Laplacian smoothing in polycube segmentation')
parser.add_argument('--normal_alignment_gamma',
                    default=1e3, type=float,
        help='Gamma for normal alignment deformation')
parser.add_argument('--normal_alignment_s',
                    default=6, type=int,
        help='Number of nearest neighbors for normal alignment deformation')
parser.add_argument('--normal_alignment_alpha',
                    default=0.5, type=float,
        help='Weight for the smoothness term in normal alignment deformation')
parser.add_argument('--normal_alignment_max_epochs',
                    default=100000, type=int,
        help='Maximum epochs for normal alignment deformation')
parser.add_argument('--normal_alignment_error_threshold',
                    default=1e-3, type=float,
        help='Error threshold for convergence in normal alignment deformation')
parser.add_argument('--normal_alignment_weights',
                    default='{"node": 1.0, "rotate_matrix": 0.0}', type=str_to_dict,
        help='Weights for different components in the normal alignment deformation')
parser.add_argument('--output_file',
                    default='./data/total_mesh.vtu', type=str,
        help='Output file for the deformed mesh')

options = vars(parser.parse_args())
bm.set_backend(options['backend'])


# 加载网格
origin_mesh = pickle.load(open("./data/unit_sphere_mesh_torch.pkl", "rb"))

for i in range(10):
    print("Iteration:", i)
    processor = PolyCubeProcessor(origin_mesh)
    # 进行法线平滑变形
    deformed_mesh = processor.mesh_normal_smooth_deformation(
        sigma=options['normal_smooth_sigma'],
        s=options['normal_smooth_s'],
        alpha=options['normal_smooth_alpha'],
        max_epochs=options['normal_smooth_max_epochs'],
        error_threshold=options['normal_smooth_error_threshold'],
        weights=options['normal_alignment_weights']
    )

    # 进行多面体分割
    segmentator = processor.polycube_segmentator(
        straighten_max_iter=options['straighten_max_iter'],
        merge_min_size=options['merge_min_size'],
        laplacian_smooth_alpha=options['laplacian_smooth_alpha'],
        laplacian_smooth_max_iter=options['laplacian_smooth_max_iter']
    )
    # 进行法线对齐变形
    normal_alignment_mesh = processor.mesh_normal_alignment_deformation(
        gamma=options['normal_alignment_gamma'],
        s=options['normal_alignment_s'],
        alpha=options['normal_alignment_alpha'],
        max_epochs=options['normal_alignment_max_epochs'],
        error_threshold=options['normal_alignment_error_threshold'],
        weights= options['normal_alignment_weights']
    )

# 导出网格
normal_alignment_mesh.to_vtk(fname="./data/total_mesh.vtu")
print("All processes are done!!")
