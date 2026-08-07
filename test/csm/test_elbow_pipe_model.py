import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, TriangleMesh
from fealpy.mesher import ElbowPipeMesher


# 测试数据 - 不同参数组合
test_parameters = [
    {
        "name": "default_parameters",
        "params": {
            "D": 25e-3,
            "bend_angle": 90.0,
            "R_bend_inner": 1.5,
            "L_in_ratio": 5.0,
            "L_out_ratio": 10.0,
            "wall_thickness": 5e-3,
        },
        "expected_nodes_min": 4000,
        "expected_cells_min": 18000
    },
    {
        "name": "small_pipe",
        "params": {
            "D": 10e-3,
            "bend_angle": 90.0,
            "R_bend_inner": 2.0,
            "L_in_ratio": 3.0,
            "L_out_ratio": 5.0,
            "wall_thickness": 2e-3,
        },
        "expected_nodes_min": 2000,
        "expected_cells_min": 8000
    }
]


class TestElbowPipeModel:
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("test_data", test_parameters)
    def test_init_mesh(self, test_data, backend):
        """测试网格初始化功能"""
        bm.set_backend(backend)
        
        # 创建弯管网格生成器
        mesher = ElbowPipeMesher(test_data["params"])
        
        # 生成网格
        mesh = mesher.init_mesh()
        
        # 验证网格类型
        assert isinstance(mesh, TetrahedronMesh), "网格类型应为TetrahedronMesh"
        
        # 验证网格尺寸
        assert mesh.number_of_nodes() >= test_data["expected_nodes_min"], \
            f"节点数量不足: {mesh.number_of_nodes()} < {test_data['expected_nodes_min']}"
        
        assert mesh.number_of_cells() >= test_data["expected_cells_min"], \
            f"单元数量不足: {mesh.number_of_cells()} < {test_data['expected_cells_min']}"
        
        # 验证几何维度
        assert mesh.geo_dimension() == 3, "几何维度应为3"
        
        # 验证区域标签存在
        assert "region" in mesh.celldata, "网格应包含区域标签"
        
        # 验证固体区域存在
        regions = mesh.celldata["region"]
        solid_region = 2  # 根据用户手册，2为固体区域
        assert solid_region in regions, "网格应包含固体区域"
        

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("test_data", [test_parameters[0]])
    def test_mesh_data(self, test_data, backend):
        """测试网格数据提取功能"""
        bm.set_backend(backend)
        
        # 创建弯管网格生成器
        mesher = ElbowPipeMesher(test_data["params"])
        
        # 获取网格数据
        mesh_data = mesher.mesh_data()
        
        # 验证必要字段存在
        required_fields = [
            "node", "tetra", "tetra_region", "boundary_tri", 
            "boundary_tri_marker", "interface_tri", "physical_name_to_dimtag"
        ]
        
        for field in required_fields:
            assert field in mesh_data, f"网格数据缺少必要字段: {field}"
        
        # 验证物理组映射
        phys_name_to_dimtag = mesh_data["physical_name_to_dimtag"]
        required_physical_groups = [
            "fluid", "solid", "inlet", "outlet", 
            "fsi_interface", "outer_wall",
            "solid_inlet_end", "solid_outlet_end"
        ]
        
        for group in required_physical_groups:
            assert group in phys_name_to_dimtag, f"缺少物理组: {group}"
        
        # 验证边界三角形
        boundary_tri = mesh_data["boundary_tri"]
        boundary_tri_marker = mesh_data["boundary_tri_marker"]
        
        assert len(boundary_tri) == len(boundary_tri_marker), \
            "边界三角形和标记数量应一致"
        
        # 验证FSI界面
        interface_tri = mesh_data["interface_tri"]
        assert len(interface_tri) > 0, "应存在FSI界面三角形"
        
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("test_data", [test_parameters[0]])
    def test_boundary_extraction(self, test_data, backend):
        """测试边界提取功能"""
        bm.set_backend(backend)
        
        # 创建弯管网格生成器
        mesher = ElbowPipeMesher(test_data["params"])
        mesh_data = mesher.mesh_data()
        
        # 提取物理组映射
        phys_name_to_dimtag = mesh_data["physical_name_to_dimtag"]
        
        # 获取固体端面标签
        solid_inlet_tag = phys_name_to_dimtag['solid_inlet_end'][1]
        solid_outlet_tag = phys_name_to_dimtag['solid_outlet_end'][1]
        
        # 获取边界三角形和标记
        boundary_tri = mesh_data["boundary_tri"]
        boundary_tri_marker = mesh_data["boundary_tri_marker"]
        
        # 计算每个物理组的三角形数量
        marker_counts = {}
        for marker in boundary_tri_marker:
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        # 验证固体端面存在边界三角形
        assert solid_inlet_tag in marker_counts, "应存在固体入口端边界"
        assert solid_outlet_tag in marker_counts, "应存在固体出口端边界"
        
        # 验证固体端面有足够数量的边界三角形
        assert marker_counts[solid_inlet_tag] > 10, "固体入口端边界三角形不足"
        assert marker_counts[solid_outlet_tag] > 10, "固体出口端边界三角形不足"
    
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("test_data", [test_parameters[0]])
    def test_vtk_export(self, test_data, backend, tmp_path):
        """测试VTK文件导出功能"""
        bm.set_backend(backend)
        
        # 创建弯管网格生成器
        mesher = ElbowPipeMesher(test_data["params"])
        
        # 生成网格和数据
        tet_mesh = mesher.init_mesh()
        mesh_data = mesher.mesh_data()
        
        # 导出体网格
        tet_path = tmp_path / "elbow_pipe_tetra.vtu"
        tet_mesh.to_vtk(fname=str(tet_path))
        assert tet_path.exists(), "体网格VTK文件未生成"
        
        # 导出边界网格
        tri_boundary = TriangleMesh(mesh_data["node"], mesh_data["boundary_tri"])
        tri_boundary_path = tmp_path / "elbow_pipe_boundary_tri.vtu"
        tri_boundary.to_vtk(fname=str(tri_boundary_path))
        assert tri_boundary_path.exists(), "边界网格VTK文件未生成"
        
        # 导出界面网格
        tri_interface = TriangleMesh(mesh_data["node"], mesh_data["interface_tri"])
        tri_interface_path = tmp_path / "elbow_pipe_interface_tri.vtu"
        tri_interface.to_vtk(fname=str(tri_interface_path))
        assert tri_interface_path.exists(), "界面网格VTK文件未生成"
    

if __name__ == "__main__":
    # 创建测试实例
    test_instance = TestElbowPipeModel()
    
    # 运行特定测试
    test_data = test_parameters[0]
    
    print("运行网格初始化测试...")
    test_instance.test_init_mesh(test_data, 'numpy')
    
    print("运行网格数据测试...")
    test_instance.test_mesh_data(test_data, 'numpy')
    
    print("运行边界提取测试...")
    test_instance.test_boundary_extraction(test_data, 'numpy')
    
    # print("运行VTK导出测试...")
    # import tempfile
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     test_instance.test_vtk_export(test_data, 'numpy', tmp_dir)
    
    print("所有测试完成!")