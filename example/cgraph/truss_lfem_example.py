import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")
WORLD_GRAPH = cgraph.WORLD_GRAPH

problem_data = cgraph.create("Truss3dData")
mesh_creator = cgraph.create("CreateMesh") 
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("BarMaterial")
truss = cgraph.create("Truss")
solver = cgraph.create("DirectSolver")
post = cgraph.create("TrussPostprocess")

problem_data()

mesh_creator(mesh_type="edge", node=problem_data().node, cell=problem_data().cell)

spacer(type="lagrange", mesh=mesh_creator().mesh, p=1)

materialer(property="Steel", bar_E=1500.0, bar_A=2000.0)

truss(space=spacer(),
      mesh=mesh_creator().mesh,
      bar_E=materialer().E,
      A=materialer().A,
      p=900.0,
      top_z=5080.0,
      fixed_nodes=[6, 7, 8, 9])

solver(A=truss().K, b=truss().F, solver="scipy", matrix_type="G")

post(uh=solver().out, mesh=mesh_creator().mesh, E=materialer().E)

WORLD_GRAPH.output(
    mesh=mesh_creator().mesh,
    u=post().u,
    strain=post().strain, 
    stress=post().stress
    )

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())