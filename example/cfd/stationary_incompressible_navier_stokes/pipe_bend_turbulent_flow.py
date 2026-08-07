from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.solver import spsolve, cg, gmres
from fealpy.backend import backend_manager as bm
from fealpy.mesher import ElbowPipeMesher
# from fealpy.mesh import TetrahedronMesh

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.1,     # 使用默认网格大小策略
    "mesh_size_bend": 0.1,
    "mesh_size_interface": 0.1,
}
mesher = ElbowPipeMesher(dim=2, params=params)


# 1. 流体单元
from fealpy.mesh import TriangleMesh

def extract_fluid_mesh(mesher):
    """
    从整体三角形网格中提取指定区域的纯净子网格
    """
    mesh = mesher.init_mesh()
    mesh_data = mesher.mesh_data()

    node = mesh_data["node"]
    tri = mesh_data["triangle"]
    tri_region = mesh_data["triangle_region"]
    name2tag = mesh_data["physical_name_to_dimtag"]

    fluid_id = name2tag["fluid"][1]
    inlet_id = name2tag["inlet"][1]
    outlet_id = name2tag["outlet"][1]
    wall_id = name2tag["outer_wall"][1]

    sub_tri_old = tri[tri_region == fluid_id]
    sub_node_id, sub_tri_new = bm.unique(sub_tri_old.reshape(-1), return_inverse=True)
    sub_node = node[sub_node_id]
    sub_tri_new = sub_tri_new.reshape(sub_tri_old.shape)
    # sub_tri_new[:, [1, 2]] = sub_tri_new[:, [2, 1]]
    sub_mesh = TriangleMesh(sub_node, sub_tri_new)
    sub_mesh.fluid_node_id = sub_node_id

    be = mesh_data["boundary_edge"]
    b_mark = mesh_data["boundary_edge_marker"]

    inlet = be[b_mark == inlet_id]
    outlet = be[b_mark == outlet_id]
    wall = be[b_mark == wall_id]
    fsi = be[b_mark == name2tag["fsi_interface"][1]]

    # 旧节点 → 新节点
    global_to_sub = -bm.ones(mesh.number_of_nodes(), dtype=bm.int64)
    global_to_sub[sub_node_id] = bm.arange(sub_node_id.shape[0], dtype=bm.int64)

    # 映射边
    be_new = global_to_sub[be]
    inlet_new = global_to_sub[inlet]
    outlet_new = global_to_sub[outlet]
    wall_new = global_to_sub[wall]
    fsi_new = global_to_sub[fsi]

    # 过滤非法边（有 -1 的）
    mask_be = bm.all(be_new >= 0, axis=1)
    mask_inlet = bm.all(inlet_new >= 0, axis=1)
    mask_outlet = bm.all(outlet_new >= 0, axis=1)
    mask_wall = bm.all(wall_new >= 0, axis=1)
    mask_fsi = bm.all(fsi_new >= 0, axis=1)
    sub_mesh.be = be_new[mask_be]
    sub_mesh.inlet = inlet_new[mask_inlet]
    sub_mesh.outlet = outlet_new[mask_outlet]
    sub_mesh.wall = wall_new[mask_wall]
    sub_mesh.fsi = fsi_new[mask_fsi]

    return sub_mesh

fluid_mesh = extract_fluid_mesh(mesher)


options = {
    'backend': 'numpy',
    'solve': 'direct',
    'method': 'Newton',
    'run': 'main',
    'maxstep': 100,
    'tol':1e-10,
    'error_com': False,
    'rho' : 1.0,
    'mu': 0.001,
    'pbar_log': True,
    'log_level': 'INFO',
    'apply_bc': 'dirichlet_dof'
}

from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.cfd.model.stationary_incompressible_navier_stokes.hydraulic_pipe_flow_model import HydraulicPipeFlowModel2D
pde = HydraulicPipeFlowModel2D(options=options, mesh=fluid_mesh)
fluid_model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh=fluid_mesh, options=options)
uspace = fluid_model.fem.uspace
u1, p1 = fluid_model.run()
fluid_mesh.nodedata["u"] = u1.reshape(2, -1).T
fluid_mesh.nodedata["p"] = p1
fluid_mesh.to_vtk("fluid.vtu")

def assembly_BForm(model, uh):
    BForm = model.fem.BForm()
    LForm = model.fem.LForm()
    model.fem.update(uh)
    A = BForm.assembly() 
    b = LForm.assembly()
    A, b = model.fem.apply_bc[model.apply_bc_str](A, b, model.pde)
    if model.equation.pressure_neumann == True:
        A, b = model.fem.lagrange_multiplier(A, b, c = model.pde.pressure_integral_target())
    return A, b

A, b = assembly_BForm(fluid_model, u1)
