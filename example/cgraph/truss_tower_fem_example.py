import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("TrussTower3d")
mesher = cgraph.create("TrussTowerMesh")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("TrussTowerMaterial")
truss_tower = cgraph.create("TrussTower")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
strain_stress = cgraph.create("BarStrainStress")

model(
    dov=0.015,
    div=0.010,
    doo=0.010,
    dio=0.007,
    load=84820.0
)
mesher(
    n_panel = 19,
    Lz = 19.0,
    Wx = 0.45,
    Wy = 0.40,
    lc = 0.1,
    ne_per_bar = 1,
    face_diag = True
)
spacer(type="lagrange", mesh=mesher(), p=1)
materialer(property="Steel", type="bar", 
    dov=model().dov,
    div=model().div,
    doo=model().doo,
    dio=model().dio,
    E=2.0e11, nu=0.3)
truss_tower(
    dov=model().dov,
    div=model().div,
    doo=model().doo,
    dio=model().dio,
    GD = model().GD,
    space = spacer(),
    E = materialer().E,
    nu = materialer().nu,
    load = model().external_load,
    dirichlet_dof = model().dirichlet_dof,
    vertical = 76,
    other = 176
)
solver(A = truss_tower().K,
       b = truss_tower().F)

postprocess(out = solver().out, node_ldof=3, type="Truss")
strain_stress(
    dov=model().dov,
    div=model().div,
    doo=model().doo,
    dio=model().dio,
    E = materialer().E,
    nu = materialer().nu,
    mesh = mesher(),
    uh = postprocess().uh,
    ele_indices = None
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(uh=postprocess().uh, strain=strain_stress().strain, stress=strain_stress().stress)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())