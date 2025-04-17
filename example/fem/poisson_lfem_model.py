
import fealpy.cgraph as fcg

from fealpy.pde.poisson_2d import CosCosData
from fealpy.cgraph.fem import PoissonFEM
from fealpy.cgraph.mesh import Box2d, Error
from fealpy.cgraph.functionspace import FunctionSpace, TensorToFEFunction
from fealpy.cgraph.solver import CGSolver


graph = fcg.WORLD_GRAPH
pde = CosCosData()

### Initialize nodes
mesher = Box2d("triangle")
mesher.set_default(nx=16, ny=16)
spacer = FunctionSpace("lagrange")
gd = fcg.Source(pde.dirichlet)
source = fcg.Source(pde.source)
poisson_fem = fcg.Sequential(
    PoissonFEM(),
    CGSolver(),
)
arr2func = TensorToFEFunction()
true_solution = fcg.Source(pde.solution)
estimator = Error()


### Construct computational graph
mesher.OUT        >> spacer.IN

spacer.OUT        >> poisson_fem.IN.space
gd.OUT            >> poisson_fem.IN.gd
source.OUT        >> poisson_fem.IN.source

poisson_fem.OUT  >> arr2func.IN.tensor
spacer.OUT        >> arr2func.IN.space

mesher.OUT        >> estimator.IN.mesh
arr2func.OUT      >> estimator.IN.u
true_solution.OUT >> estimator.IN.v

graph.OutputPort("error") << estimator.OUT

### Execute
if __name__ == "__main__":
    result = graph.execute()
    print("Error: ", result['error'])
