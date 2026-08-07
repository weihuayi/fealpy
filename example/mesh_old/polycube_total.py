from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mesh_deformation import MeshNormalSmoothDeformation, MeshNormalAlignmentDeformation
from polycube_processor import PolyCubeProcessor


if __name__ == "__main__":
    # 加载网格
    origin_mesh = pickle.load(open("unit_sphere_mesh_torch.pkl", "rb"))

    for i in range(10):
        # 法向光滑变形
        normal_smooth_weights = {"node": 1.0, "rotate_matrix": 0.0}
        normal_smooth_deformer = MeshNormalSmoothDeformation(origin_mesh,
                                               sigma=0.1, s=7, alpha=0.5, max_epochs=100000,
                                               error_threshold=1e-3, weights=normal_smooth_weights)
        normal_smooth_mesh = normal_smooth_deformer.optimize()
        # normal_smooth_deformer.plot_mesh(normal_smooth_mesh, title="Normal Smooth Mesh")

        # 面朝向分区
        polycube_processor = PolyCubeProcessor(normal_smooth_mesh)
        polycube_processor.build_candidate_charts()
        # v_mesh = polycube_processor.volume_mesh
        # v_mesh.to_vtk(fname='v_mesh.vtu')
        # bd_mesh = polycube_processor.surface_mesh
        # bd_mesh.celldata['initial_labels'] = bm.copy(polycube_processor.labels)
        # bd_mesh.celldata['charts'] = bm.copy(polycube_processor.face2chart)
        # bd_mesh.to_vtk(fname='bd_mesh.vtu')
        polycube_processor.extract_candidate_edges_vertices()
        polycube_processor.straighten_edges(max_iter=5)
        polycube_processor.merge_small_charts(min_size=5)
        is_valid = polycube_processor.validate_topology()
        print(f"Topology is valid: {is_valid}")
        polycube_processor.laplacian_smooth(alpha=0.2, max_iter=3)
        polycube_processor.edge_projection()
        turning_points = polycube_processor.detect_turning_points()
        print("turning points:", turning_points)
        face_processed_mesh = polycube_processor.volume_mesh
        # normal_smooth_deformer.plot_mesh(face_processed_mesh, title="Face Processed Mesh")
        bd_mesh = polycube_processor.surface_mesh
        bd_mesh.celldata['initial_labels'] = bm.copy(polycube_processor.labels)
        bd_mesh.celldata['charts'] = bm.copy(polycube_processor.face2chart)
        bd_mesh.to_vtk(fname=f'face_processed_{i}.vtu')

        # 法向对齐变形
        normal_alignment_weights = {"node": 1.0, "rotate_matrix": 0.0}
        normal_alignment_deformer = MeshNormalAlignmentDeformation(face_processed_mesh,
                                                  gamma=1e3, s=6, alpha=0.5, max_epochs=100000,
                                                  error_threshold=1e-3, weights=normal_alignment_weights)
        normal_alignment_mesh = normal_alignment_deformer.optimize()
        # normal_alignment_deformer.plot_mesh(normal_alignment_mesh, title="Normal Alignment Mesh")

        origin_mesh = normal_alignment_mesh

    # 导出网格
    normal_alignment_mesh.to_vtk(fname="total_mesh.vtu")
    print("All processes are done!!")
