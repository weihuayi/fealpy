import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model942 = cgraph.create("Bar942Data")
mesher942 = cgraph.create("Bar942Mesh")
spacer = cgraph.create("FunctionSpace")
materialer942 = cgraph.create("Bar942Material")
bar942_model = cgraph.create("Bar942Model")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
strain_stress = cgraph.create("Bar942StrainStress")

model942()
mesher942(d1 = 2135,
        d2 = 5335,
        d3 = 7470,
        d4 = 9605,
        r2 = 4265,
        r3 = 6400,
        r4 = 8535,
        l3 = 43890,
        l2 = None,
        l1 = None
        )
spacer(type="lagrange", mesh=mesher942(), p=1)
materialer942(property="structural-steel", type="bar", E=2.1e5, nu=0.3)

bar942_model(
    GD = model942().GD,
    space = spacer(),
    E = materialer942().E,
    nu = materialer942().nu,
    external_load = model942().external_load,
    dirichlet_dof = model942().dirichlet_dof,
    dirichlet_bc = model942().dirichlet_bc,
    penalty = 1e12
)

solver(A = bar942_model().K,
       b = bar942_model().F)

postprocess(out = solver().out, node_ldof=3, type="Truss")
coord(mesh=mesher942(), vref=None, index=None)

strain_stress(
    E = materialer942().E,
    nu = materialer942().nu,
    mesh = mesher942(),
    uh = solver().out,
    coord_transform = coord().R
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(model=model942(),mesh=mesher942())
WORLD_GRAPH.output(uh=postprocess().uh, 
                   strain=strain_stress().strain, stress=strain_stress().stress
                   )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())