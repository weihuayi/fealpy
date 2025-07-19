from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.polycube import PolyCubeProcessor

bm.set_backend("pytorch")



if __name__ == "__main__":
    init_tet_mesh = TetrahedronMesh.from_vtu('data/bunny_opt.vtu')

    for i in range(10):
        print("Iteration:", i)
        processor = PolyCubeProcessor(init_tet_mesh)
        # 进行法线平滑变形
        deformed_mesh = processor.mesh_normal_smooth_deformation(
            sigma=0.1,
            s=1,
            alpha=0.5,
            max_epochs=100000,
            error_threshold=1e-3,
            weights={"node": 1.0, "rotate_matrix": 0.0}
        )

        # 进行多面体分割
        segmentator = processor.polycube_segmentator(
            straighten_max_iter=5,
            merge_min_size=5,
            laplacian_smooth_alpha=0.3,
            laplacian_smooth_max_iter=3
        )
        # 进行法线对齐变形
        normal_alignment_mesh = processor.mesh_normal_alignment_deformation(
            gamma=1e3,
            s=1,
            alpha=0.5,
            max_epochs=100000,
            error_threshold=1e-3,
            weights={"node": 1.0, "rotate_matrix": 0.0}
        )
        init_tet_mesh = normal_alignment_mesh
    # 导出网格
    init_tet_mesh.to_vtk(fname="./data/polycube_bunny.vtu")
    bd_mesh = processor.segmentator.surface_mesh
    bd_mesh.celldata['chart'] = bm.copy(processor.segmentator.face2chart)
    bd_mesh.to_vtk(fname="./data/polycube_bunny_bd.vtu")


    print("All processes are done!!")