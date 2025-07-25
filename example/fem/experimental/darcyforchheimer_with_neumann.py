from fealpy.backend import backend_manager as bm

#from fealpy.fem import NeumannBC
from fealpy.mesh import TriangleMesh
from fealpy.solver import spsolve
from fealpy.utils import timer
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import LinearForm, SourceIntegrator, ScalarSourceIntegrator
from fealpy.fem import ScalarNeumannBCIntegrator
from fealpy.fem import BilinearForm, GradPressureIntegrator, ScalarMassIntegrator 
from fealpy.fem import BlockForm, LinearBlockForm
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC

from fealpy.solver import spsolve, cg, gmres
from fealpy.model.darcyforchheimer.cos_cos_data_2d import CosCosData2D

bm.set_backend('numpy')
udegree = 0
pdegree = 1

maxstep=1000

q = 4
pde = CosCosData2D()
domain = pde.domain()

mesh = TriangleMesh.from_box(domain, nx=40, ny=40)

pspace = LagrangeFESpace(mesh, p=pdegree)
space = LagrangeFESpace(mesh, p=udegree, ctype='D')
uspace = TensorFunctionSpace(space,(2,-1))

uh = uspace.function()
ph = pspace.function()
u0 = uspace.function()
p0 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = ugdof + pgdof


# BilinearForm
u_bform = BilinearForm(uspace)
M = ScalarMassIntegrator(coef=pde.mu, q=q)
u_bform.add_integrator(M)

p_bform = BilinearForm(pspace,uspace)
D = GradPressureIntegrator(q=q)
p_bform.add_integrator(D)


### assemly the right term
ulform = LinearForm(uspace)
f = ScalarSourceIntegrator(pde.f, q=q)
ulform.add_integrator(f)

plform = LinearForm(pspace)
g = ScalarSourceIntegrator(pde.g, q=q)
gn = ScalarNeumannBCIntegrator(source=pde.neumann, q=q)
plform.add_integrator(g)
plform.add_integrator(gn)
bform = LinearBlockForm([ulform, plform])

def u_norm_coef(bcs, index, uh0=u0):
        return pde.beta*bm.sqrt(uh0(bcs, index)**2)
    

for i in range(maxstep):
    m = bm.sum(u0[:] - uh[:])
    ## BilinearForm
    Mu = ScalarMassIntegrator(coef=u_norm_coef, q=q)
    u_bform.add_integrator(Mu)
    
    BForm = BlockForm([[u_bform, p_bform],
                    [p_bform.T, None]])

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
    ph[:] = ph[:] - ip1
            
    
    
    
    res_u = mesh.error(u0, uh)
    res_p = mesh.error(p0, ph)
    print('res_u', res_u)
    print('res_p', res_p)
    
    if res_u + res_p < 1e-5:
        print(i+1)
        break
    u0[:] = uh
    p0[:] = ph
    
eu = mesh.error(pde.velocity, uh)
ep = mesh.error(pde.pressure, ph)
    
print('eu', eu)
print('ep', ep)




