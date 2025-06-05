from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.simulation.fem import IPCS
from fealpy.cfd.problem.IncompressibleNS import Channel

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace

#backend = 'pytorch'
backend = 'numpy'
bm.set_backend(backend)

pde = Channel()
ns_eq = IncompressibleNS(pde)
fem = IPCS(ns_eq)
sim = fem.simulation()
sim.run()

exit()
from fealpy.solver import spsolve
sim.params.set_solver(spsolve)
#sim.params.set_solver(solver_type='direct', api='scipy')

print(ns_eq)
print(fem.params)
print(sim.params)


print(sim.params)
fem.set.assembly(quadrature_order=4)
space = LagrangeFESpace(pde.mesh, p=3)
uspace = TensorFunctionSpace(space, (2,-1))
fem.set.uspace(space=uspace)
fem.set.uspace('Lagrange', p=3)
fem.set.pspace(space)
print(fem.params)
