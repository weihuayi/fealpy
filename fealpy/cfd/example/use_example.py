from fealpy.cfd.equation import IncompressibleNS
from fealpy.cfd.simulation.fem import IPCS
from fealpy.cfd.problem.IncompressibleNS import Channel

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace

backend = 'numpy'
bm.set_backend(backend)

pde = Channel()
ns_eq = IncompressibleNS(pde)
fem = IPCS(ns_eq)
sim = fem.simulation()
sim.run()





'''
from fealpy.solver import spsolve
sim.params.set_solver(spsolve)
'''
'''
u1, p1 = sim.one_step(u0, p0)
print(bm.sum(bm.abs(u1)))
print(bm.sum(bm.abs(p1)))
'''

#print(fem.params)
#print(ns_eq)
#print(sim.params)
'''
discret.set.assembly(quadrature_order=4)
space = LagrangeFESpace(pde.mesh, p=3)
space = TensorFunctionSpace(space, (2,-1))
discret.set.uspace(space=space)
discret.set.uspace('Lagrange', p=3)
discret.set.pspace(space)
'''

