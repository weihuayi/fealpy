import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("TrussTower3d")
mesher = cgraph.create("TrussTowerMesh")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("BarMaterial")
truss_tower = cgraph.create("TrussTower")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
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
materialer(property="Steel", bar_type="truss_tower", E=2.0e11, nu=0.3)
truss_tower(
    dov=model().dov,
    div=model().div,
    doo=model().doo,
    dio=model().dio,
    space_type="lagrangespace",
    GD = model().GD,
    mesh = mesher(),
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
coord(mesh=mesher(), index=None)
strain_stress(
    bar_type="truss_tower",
    E = materialer().E,
    nu = materialer().nu,
    mesh = mesher(),
    uh = solver().out,
    coord_transform = coord().R
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(uh=postprocess().uh, strain=strain_stress().strain, stress=strain_stress().stress
                )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())