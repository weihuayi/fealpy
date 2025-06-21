import numpy as np
import matplotlib.pyplot as plt

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm, BilinearForm
from fealpy.decorator import barycentric, cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace 
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

def test_bilinear_operator(u, f, o):
    ns = 4
    q = 5
    maxit = 6
    mesh = TriangleMesh.from_unit_square(nx=ns, ny=ns)
     
    errorType = ['$|| u - Ou||_{\Omega,0}$']
    errorMatrix = np.zeros((1, maxit), dtype=np.float64)
    NDof = np.zeros(maxit, dtype=np.int_)
    
    for i in range(maxit):
        print("The {}-th computation:".format(i))
        space = LagrangeFESpace(mesh, p=1, doforder='sdofs')
        NDof[i] = space.number_of_global_dofs()
        x = space.interpolate(u)

        LForm = LinearForm(space)
        LForm.add_domain_integrator(ScalarSourceIntegrator(f=f,q=q))
        b = LForm.assembly()

        BLForm = BilinearForm(space)
        BLForm.add_domain_integrator(o)
        A = BLForm.assembly()
        
        fun0 = space.function(array=A@x)
        fun1 = space.function(array=b)
        L2error = mesh.error(fun0,fun1,q=5)
        maxerror = np.max(A@x-b)

        errorMatrix[0, i] = maxerror
        print(maxerror) 
        if i < maxit-1:
            mesh.uniform_refine()

    showmultirate(plt, 1, NDof, errorMatrix,  errorType, propsize=20)
    #show_error_table(NDof, errorType, errorMatrix)
    plt.show()

@cartesian
def source(p): 
    x = p[...,0]
    y = p[...,1]
    result = np.zeros_like(x)
    return result

@cartesian
def function(p):
    x = p[...,0]
    y = p[...,1]
    #result = np.sin(x)*np.cos(y)
    result = x+y
    return result


integrator = ScalarDiffusionIntegrator(q=5)
test_bilinear_operator(function, source, integrator)
