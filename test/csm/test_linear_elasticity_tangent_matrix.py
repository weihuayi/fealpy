import numpy as np
import matplotlib.pyplot as plt
import ipdb

import pytest

from fealpy.csm import LinearElasticity
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace as Space
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator

from fealpy.functionspace import LagrangeFiniteElementSpace as OldSpace
from fealpy.boundarycondition import DirichletBC as OldDirichletBC
from fealpy.boundarycondition import NeumannBC as OldNeumannBC


def test_linear_elasticity_2d(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData2d

    pde = BoxDomainData2d()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box=domain, nx=n, ny=n)
    
    GD = mesh.geo_dimension()
    le = LinearElasticity(mesh, pde)
    D = le.tangent_matrix()
    
    # 新接口程序
    # 构建双线性型，表示问题的微分形式
    space = Space(mesh, p=p, doforder='vdims')
    uh = space.function(dim=GD)
    vspace = GD*(space, ) # 把标量空间张成向量空间
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()

    A = bform.get_matrix()
    
    ubform = BilinearForm(vspace)
    integrator = ProvidesSymmetricTangentOperatorIntegrator(D, q=4)
    ubform.add_domain_integrator(integrator)
    A0 = ubform.assembly()
    print('A:', A)
    print('A0:', A0)
    print(np.max(A-A0))

def test_linear_elasticity_3d(p, n):
    """
    @brief Lagrange 元求解线弹性问题
    """
    from fealpy.pde.linear_elasticity_model import BoxDomainData3d

    pde = BoxDomainData3d()
    domain = pde.domain()
    mesh = pde.init_mesh(n=n)
    le = LinearElasticity(mesh, pde)
    D = le.tangent_matrix()
    print(D)



if __name__ == "__main__":
    test_linear_elasticity_2d(1, 10)
    #test_linear_elasticity_3d(1, 10)
