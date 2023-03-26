import numpy as np
import pytest

from fealpy.fem import BilinearForm


def test_truss():

    from fealpy.mesh import EdgeMesh
    from fealpy.functionspace import LagrangeFiniteElementSpaceOnEdgeMesh as Space
    from fealpy.fem import TrussStructureIntegrator
    
    mesh = EdgeMesh.from_tower()
    GD = mesh.geo_dimension()
    space = Space(mesh, p=1)

    bform = BilinearForm(GD*(space,))
    bform.add_domain_integrator(TrussStructureIntegrator(1500, 2000))

    bform.assembly()
    print(bform.M.toarray())


if __name__ == '__main__':
    test_truss()



