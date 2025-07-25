from fealpy.backend import backend_manager as bm

#from fealpy.fem import NeumannBC
from fealpy.mesh import TriangleMesh
from fealpy.solver import spsolve
from fealpy.utils import timer
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import LinearForm, SourceIntegrator, ScalarSourceIntegrator, VectorSourceIntegrator
from fealpy.fem import ScalarNeumannBCIntegrator
from fealpy.fem import BilinearForm, GradPressureIntegrator, ScalarMassIntegrator 
from fealpy.fem import BlockForm, LinearBlockForm
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC

from fealpy.solver import spsolve, cg, gmres
from fealpy.model.darcy.cos_cos_data_2d import CosCosData2D

bm.set_backend('numpy')
udegree = 0
pdegree = 1

q = 4
pde = CosCosData2D()
domain = pde.domain()

mesh = TriangleMesh.from_box(domain, nx=80, ny=80)

pspace = LagrangeFESpace(mesh, p=pdegree)
space = LagrangeFESpace(mesh, p=udegree, ctype='D')
uspace = TensorFunctionSpace(space,(2,-1))

uh = uspace.function()
ph = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = ugdof + pgdof

# BilinearForm
u_bform = BilinearForm(uspace)
M = ScalarMassIntegrator(coef=pde.mu_coef, q=q)
u_bform.add_integrator(M)

p_bform = BilinearForm(pspace,uspace)
D = GradPressureIntegrator(q=q)
p_bform.add_integrator(D)

BForm = BlockForm([[u_bform, p_bform],
                   [p_bform.T, None]])

### assemly the right term
ulform = LinearForm(uspace)
f = ScalarSourceIntegrator(pde.f, q=q)
ulform.add_integrator(f)

plform = LinearForm(pspace)
g = ScalarSourceIntegrator(pde.g, q=q)
gn = ScalarNeumannBCIntegrator(source=pde.neumann, q=q)
plform.add_integrator([gn])

bform = LinearBlockForm([ulform, plform])
b = bform.assembly()
A = BForm.assembly()

# Modify matrix
threshold = bm.zeros(gdof, dtype=bm.bool)
threshold[ugdof] = True
gd = bm.zeros(gdof)
gd[ugdof] = 1
BC = DirichletBC((uspace,pspace), gd=gd, 
                threshold=threshold, method='interp')

A,b = BC.apply(A,b)
x = spsolve(A, b, solver='scipy')
uh[:] = x[:ugdof]
ph[:] = x[ugdof:]
    
ip1 = mesh.integral(ph)/(sum(mesh.entity_measure('cell')))
#ph[:] = ph[:] - ip1 + pspace.integralalg.integral(pde.pressure)/(sum(mesh.entity_measure('cell')))
ph[:] = ph[:] -  ip1
            
    
    
    
eu = mesh.error(pde.velocity, uh)
ep = mesh.error(pde.pressure, ph)
print('eu', eu)
print('ep', ep)




