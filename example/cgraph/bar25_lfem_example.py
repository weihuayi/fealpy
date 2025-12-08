import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model25 = cgraph.create("BarData")
mesher25 = cgraph.create("Bar25Mesh")
spacer = cgraph.create("FunctionSpace")
materialer25 = cgraph.create("BarMaterial")
bar25_model = cgraph.create("BarModel")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
strain_stress = cgraph.create("BarStrainStress")

model25(bar_type="bar25")
mesher25()
spacer(type="lagrange", mesh=mesher25(), p=1)
materialer25(property="structural-steel", bar_type="bar25", E=1500, nu=0.3)

bar25_model(
    bar_type="bar25",
    space_type="lagrangespace",
    GD = model25().GD,
    mesh = mesher25(),
    E = materialer25().E,
    nu = materialer25().nu,
    external_load = model25().external_load,
    dirichlet_dof = model25().dirichlet_dof,
    dirichlet_bc = model25().dirichlet_bc
)

solver(A = bar25_model().K,
       b = bar25_model().F)

postprocess(out = solver().out, node_ldof=3, type="Truss")
coord(mesh=mesher25(), vref=None, index=None)

strain_stress(
    bar_type="bar25",
    E = materialer25().E,
    nu = materialer25().nu,
    mesh = mesher25(),
    uh = solver().out,
    coord_transform = coord().R
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(model=model25(), mesh1=mesher25())
WORLD_GRAPH.output(uh=postprocess().uh, 
                   strain=strain_stress().strain, stress=strain_stress().stress
                   )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())