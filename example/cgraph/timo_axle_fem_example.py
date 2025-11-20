import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("Timoaxle3d")
mesher = cgraph.create("CreateMesh")
beam_materialer = cgraph.create("TimoMaterial")
axle_materialer = cgraph.create("AxleMaterial")
spacer = cgraph.create("FunctionSpace")
#spacer = cgraph.create("TensorFunctionSpace")
timoaxle_model = cgraph.create("Timoaxle")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
report = cgraph.create("SolidReport")

# 连接节点
node = bm.array([[0.0, 0.0, 0.0], [70.5, 0.0, 0.0], [141.0, 0.0, 0.0], [155.0, 0.0, 0.0],
            [169.0, 0.0, 0.0], [213.25, 0.0, 0.0], [257.5, 0.0, 0.0], [301.75, 0.0, 0.0],
            [346.0, 0.0, 0.0], [480.0, 0.0, 0.0], [614.0, 0.0, 0.0], [853.0, 0.0, 0.0],
            [1092.0, 0.0, 0.0], [1334.0, 0.0, 0.0], [1576.0, 0.0, 0.0], [1620.25, 0.0, 0.0],
            [1664.5, 0.0, 0.0], [1708.75, 0.0, 0.0], [1753.0, 0.0, 0.0], [1767.0, 0.0, 0.0],
            [1781.0, 0.0, 0.0], [1851.5, 0.0, 0.0],[1922.0, 0.0, 0.0], [169.0, 0.0, -100.0],
            [213.25, 0.0, -100.0], [257.5, 0.0, -100.0], [301.75, 0.0, -100.0], [346.0, 0.0, -100.0],
            [1576.0, 0.0, -100.0], [1620.25, 0.0, -100.0], [1664.5, 0.0, -100.0], 
            [1708.75, 0.0, -100.0], [1753.0, 0.0, -100.0]], dtype=bm.float64)

cell = bm.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8],[8, 9], [9, 10], [10, 11],
        [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
        [16, 17],[17, 18], [18, 19], [19, 20], [20, 21], 
        [21, 22], [4, 23], [5, 24], [6, 25], [7, 26], [8, 27], 
        [14, 28], [15, 29],[16, 30], [17, 31], [18,32]], dtype=bm.int32)

mesher(node = node, cell = cell)
spacer(type="lagrange", mesh=mesher(), p=1)
#spacer(type="lagrange", mesh=mesher(), p=1, gd=6)
beam_materialer(property="Steel", beam_type="Timoshemko", beam_E=2.1e11, beam_nu=0.3)
axle_materialer(property="Steel", axle_type="Spring", axle_E=1.976e6, axle_nu=-0.5)

timoaxle_model(
    space = spacer(),
    beam_E = beam_materialer().E,
    beam_mu = beam_materialer().mu,
    Ax =  beam_materialer().Ax,
    Ay = beam_materialer().Ay,
    Az = beam_materialer().Az,
    J = beam_materialer().J,
    Iy = beam_materialer().Iy,
    Iz = beam_materialer().Iz,
    axle_E = axle_materialer().E,
    axle_mu = axle_materialer().mu,
    cindex = 32,
    external_load = model().external_load,
    dirichlet_idx = model().dirichlet_dof_index,
    penalty = 1e20,
    boundary_type = "force",
    load_type = "fixed"  ,
)

solver(A = timoaxle_model().K,
       b = timoaxle_model().F)

postprocess(out = solver().out, node_ldof=6, type="Timo_beam")
report(
    path = r"C:\Users\Administrator\Desktop",
    beam_para = model().beam_para,
    axle_para = model().axle_para,
    section_shapes = "circular",
    shear_factors = 10/9,
    mesh=mesher(), 
    property="Steel",
    beam_E = beam_materialer().E,
    beam_mu = beam_materialer().mu,
    axle_E = axle_materialer().E,
    axle_mu = axle_materialer().mu,
    uh = solver().out
       )


# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher(), u=solver().out, uh=postprocess().uh, theta=postprocess().theta, report=report())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())