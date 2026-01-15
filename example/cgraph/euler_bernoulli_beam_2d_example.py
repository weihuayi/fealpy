import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("Beam2d")
mesher = cgraph.create("CreateMesh")
beam_materialer = cgraph.create("BeamMaterial")
spacer = cgraph.create("FunctionSpace")
beam_model = cgraph.create("Beam")
dbc = cgraph.create("StructuralDirichletBC")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")

# 连接节点
node = bm.array([[0], [5],[7.5]], dtype=bm.float64)
cell = bm.array([[0, 1],[1,2]] , dtype=bm.int32)

mesher(node = node, cell = cell)
spacer(type="lagrange", mesh=mesher(), p=1)

beam_materialer(property="Steel", beam_type="Euler-Bernoulli beam", beam_E=200e9, beam_nu=0.3, I=118.6e-6)

beam_model(
    space = spacer(),
    beam_E = beam_materialer().E,
    beam_nu = beam_materialer().nu,
    I = beam_materialer().I,
    distributed_load = model().f,
    beam_type = "euler_bernoulli_2d",
)

dbc(
    gd=model().dirichlet,
    isDDof=model().dirichlet_dof_index,
    K = beam_model().K,
    F=beam_model().F,
    space=spacer()
)

solver(A = dbc().K,
       b = dbc().F)

postprocess(out = solver().out, node_ldof=2, type="Euler_beam")


# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher(), u=solver().out, uh=postprocess().uh, theta=postprocess().theta)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())