# test_yee_uniform_mesher.py
import sys
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh
from fealpy.cem.mesh.yee_uniform_mesher import YeeUniformMesher

def test_2d_mesh():
    """测试2D网格功能"""
    print("=== Testing 2D Mesh ===")
    
    # Create a 2D mesh
    mesh = YeeUniformMesher(domain=(0, 1, 0, 1), nx=4, ny=4)
    nx, ny = mesh.nx, mesh.ny

    print("TD:", mesh.TD)
    print("nx, ny:", nx, ny)
    print("h:", mesh.h)
    print("origin:", mesh.origin)

    # Test coordinate properties
    print("\n--- Testing Coordinate Properties ---")
    print("node_coords shape:", mesh.node_coords.shape)
    print("edgex_coords shape:", mesh.edgex_coords.shape)
    print("edgey_coords shape:", mesh.edgey_coords.shape)
    print("cell_coords shape:", mesh.cell_coords.shape)

    # Test get_field_matrix shapes
    print("\n--- Testing Field Matrix Shapes ---")
    node_shape = mesh.get_field_matrix(etype="node").shape
    print("node shape:", node_shape)
    assert node_shape == (nx + 1, ny + 1)

    cell_shape = mesh.get_field_matrix(etype="cell").shape
    print("cell shape:", cell_shape)
    assert cell_shape == (nx, ny)

    edge_x_shape = mesh.get_field_matrix(etype="edge", axis="x").shape
    edge_y_shape = mesh.get_field_matrix(etype="edge", axis="y").shape
    print("edge x shape:", edge_x_shape)
    print("edge y shape:", edge_y_shape)
    assert edge_x_shape == (nx + 1, ny)
    assert edge_y_shape == (nx, ny + 1)

    # Test init_field_matrix mapping
    print("\n--- Testing Field Initialization ---")
    E_node = mesh.init_field_matrix(type="E")
    H_edge_x = mesh.init_field_matrix(type="H", axis="x")
    H_edge_y = mesh.init_field_matrix(type="H", axis="y")
    
    print("E node shape:", E_node.shape)
    print("H edge x shape:", H_edge_x.shape)
    print("H edge y shape:", H_edge_y.shape)
    
    assert E_node.shape == node_shape
    assert H_edge_x.shape == edge_x_shape
    assert H_edge_y.shape == edge_y_shape

    # Test location methods
    print("\n--- Testing Location Methods ---")
    test_points = bm.array([[0.25, 0.25], [0.75, 0.75]])
    
    cell_indices = mesh.cell_location(test_points)
    print("Cell indices:", [idx.shape for idx in cell_indices])
    
    node_indices = mesh.node_location(test_points)
    print("Node indices:", [idx.shape for idx in node_indices])


    # Test field initialization with functions
    print("\n--- Testing Field Initialization with Functions ---")
    
    def Ez_func(X, Y, t=0.0):
        return bm.sin(X + Y + t)

    def Hx_func(X, Y, t=0.0):
        return bm.cos(X - Y + t)

    def Hy_func(X, Y, t=0.0):
        return 0.1 * X

    # Test single field initialization
    Ez_arr = mesh.initialize_field("E_z", Ez_func, dt=0.01, times=None)
    print("E_z shape:", Ez_arr.shape)
    assert Ez_arr.shape == node_shape

    Hx_arr = mesh.initialize_field("H_x", Hx_func, dt=0.01, times=None)
    print("H_x shape:", Hx_arr.shape)
    assert Hx_arr.shape == edge_x_shape

    Hy_arr = mesh.initialize_field("H_y", Hy_func, dt=0.01, times=None)
    print("H_y shape:", Hy_arr.shape)
    assert Hy_arr.shape == edge_y_shape

    # Test with time series
    print("\n--- Testing Time Series ---")
    Ez_arr, Ez_series = mesh.initialize_field("E_z", Ez_func, dt=0.01, times=3)
    print("E_z series shape:", Ez_series.shape)
    assert Ez_series.shape[0] == 4  # times=3 gives 4 time steps (0,1,2,3)
    assert Ez_series.shape[1:] == node_shape

    # Test _init_fields_dict
    print("\n--- Testing Fields Dictionary ---")
    fields, data = mesh._init_fields_dict("E", ["z"], num_frames=5, axis_type=True)
    print("Fields keys:", fields.keys())
    print("Data keys:", data.keys())
    
    assert "z" in fields
    assert "z" in data
    
    buf = data["z"]
    print("Buffer shape:", buf.shape)
    assert buf.shape[0] == 5
    assert buf.shape[1:] == node_shape

    # Verify buffer initialization
    if isinstance(fields['z'], tuple):
        f0 = fields['z'][0]
    else:
        f0 = fields['z']
    
    diff = bm.max(bm.abs(buf[0] - f0))
    print("Max diff buf[0] vs f0:", float(diff))
    assert float(diff) == 0.0

    print("✓ All 2D tests passed!\n")

def test_3d_mesh():
    """测试3D网格功能"""
    print("=== Testing 3D Mesh ===")
    
    # Create a 3D mesh
    mesh = YeeUniformMesher(domain=(0, 1, 0, 1, 0, 1), nx=3, ny=3, nz=3)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

    print("TD:", mesh.TD)
    print("nx, ny, nz:", nx, ny, nz)

    # Test coordinate properties
    print("\n--- Testing 3D Coordinate Properties ---")
    print("node_coords shape:", mesh.node_coords.shape)
    print("edgex_coords shape:", mesh.edgex_coords.shape)
    print("edgey_coords shape:", mesh.edgey_coords.shape)
    print("edgez_coords shape:", mesh.edgez_coords.shape)
    print("facex_coords shape:", mesh.facex_coords.shape)
    print("facey_coords shape:", mesh.facey_coords.shape)
    print("facez_coords shape:", mesh.facez_coords.shape)
    print("cell_coords shape:", mesh.cell_coords.shape)

    # Test get_field_matrix shapes
    print("\n--- Testing 3D Field Matrix Shapes ---")
    node_shape = mesh.get_field_matrix(etype="node").shape
    cell_shape = mesh.get_field_matrix(etype="cell").shape
    facex_shape = mesh.get_field_matrix(etype="face", axis="x").shape
    facey_shape = mesh.get_field_matrix(etype="face", axis="y").shape
    facez_shape = mesh.get_field_matrix(etype="face", axis="z").shape
    edgex_shape = mesh.get_field_matrix(etype="edge", axis="x").shape
    edgey_shape = mesh.get_field_matrix(etype="edge", axis="y").shape
    edgez_shape = mesh.get_field_matrix(etype="edge", axis="z").shape

    print("node shape:", node_shape)
    print("cell shape:", cell_shape)
    print("face x shape:", facex_shape)
    print("face y shape:", facey_shape)
    print("face z shape:", facez_shape)
    print("edge x shape:", edgex_shape)
    print("edge y shape:", edgey_shape)
    print("edge z shape:", edgez_shape)

    assert node_shape == (nx + 1, ny + 1, nz + 1)
    assert cell_shape == (nx, ny, nz)
    assert facex_shape == (nx + 1, ny, nz)
    assert facey_shape == (nx, ny + 1, nz)
    assert facez_shape == (nx, ny, nz + 1)
    assert edgex_shape == (nx, ny + 1, nz + 1)
    assert edgey_shape == (nx + 1, ny, nz + 1)
    assert edgez_shape == (nx + 1, ny + 1, nz)

    # Test init_field_matrix for 3D
    print("\n--- Testing 3D Field Initialization ---")
    E_edge_x = mesh.init_field_matrix(type="E", axis="x")
    E_edge_y = mesh.init_field_matrix(type="E", axis="y")
    E_edge_z = mesh.init_field_matrix(type="E", axis="z")
    H_face_x = mesh.init_field_matrix(type="H", axis="x")
    H_face_y = mesh.init_field_matrix(type="H", axis="y")
    H_face_z = mesh.init_field_matrix(type="H", axis="z")

    print("E edge x shape:", E_edge_x.shape)
    print("E edge y shape:", E_edge_y.shape)
    print("E edge z shape:", E_edge_z.shape)
    print("H face x shape:", H_face_x.shape)
    print("H face y shape:", H_face_y.shape)
    print("H face z shape:", H_face_z.shape)

    assert E_edge_x.shape == edgex_shape
    assert E_edge_y.shape == edgey_shape
    assert E_edge_z.shape == edgez_shape
    assert H_face_x.shape == facex_shape
    assert H_face_y.shape == facey_shape
    assert H_face_z.shape == facez_shape

    print("✓ All 3D tests passed!\n")

def test_error_handling():
    """测试错误处理"""
    print("=== Testing Error Handling ===")
    
    mesh = YeeUniformMesher(domain=(0, 1, 0, 1), nx=4, ny=4)
    
    # Test invalid field names
    try:
        mesh.initialize_field("invalid", lambda x, y: x + y, dt=0.01)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly caught invalid field name:", str(e))
    
    # Test invalid interpolation type
    try:
        mesh.interpolation(lambda x, y: x + y, "invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly caught invalid interpolation type:", str(e))
    
    # Test 3D-only properties in 2D
    try:
        _ = mesh.edgez_coords
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly caught 3D-only property access in 2D:", str(e))
    
    print("✓ All error handling tests passed!\n")

if __name__ == "__main__":
    # Set backend for testing
    bm.set_backend('numpy')  # or 'torch', 'paddle', etc.
    
    test_2d_mesh()
    test_3d_mesh() 
    test_error_handling()
    
    print("All tests completed successfully!")