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

spacer(type="lagrange", mesh=problem_data().mesh, p=1)
materialer(property="Steel", bar_E=1500.0, bar_A=2000.0)

truss(space=spacer(),
      bar_E=materialer().E,
      A=materialer().A,
      external_load=problem_data().external_load, 
      dirichlet_idx=problem_data().dirichlet_idx) 

solver(A=truss().K, b=truss().F, solver="scipy", matrix_type="G")

post(uh=solver().out, mesh=problem_data().mesh, E=materialer().E)

WORLD_GRAPH.output(mesh=problem_data().mesh,
                   u=solver().out, 
                   uh=post().uh, 
                   strain=post().strain, 
                   stress=post().stress)

WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())