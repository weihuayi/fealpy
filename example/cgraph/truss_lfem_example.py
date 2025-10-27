import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")
WORLD_GRAPH = cgraph.WORLD_GRAPH


problem_data = cgraph.create("Truss3dData")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("BarMaterial")
truss = cgraph.create("Truss")
solver = cgraph.create("DirectSolver")
post = cgraph.create("StrainStressPostprocess")


problem_data(p=900.0, top_z=5080.0, fixed_nodes=[6, 7, 8, 9])

spacer(type="lagrange", mesh=problem_data().mesh, p=1)
materialer(property="Steel", bar_E=1500.0, bar_A=2000.0)

truss(space=spacer(),
      bar_E=materialer().E,
      A=materialer().A,

      F=problem_data().F,
      fixed_dofs=problem_data().fixed_dofs)

solver(A=truss().K, b=truss().F, solver="scipy", matrix_type="G")
post(uh=solver().out, mesh=problem_data().mesh, E=materialer().E)

WORLD_GRAPH.output(
    mesh=problem_data().mesh,
    uh=post().uh,    
    strain=post().strain, 
    stress=post().stress
    )

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())