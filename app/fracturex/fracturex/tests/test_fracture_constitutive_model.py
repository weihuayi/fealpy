import ipdb

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.material.material_base import MaterialBase

from ..phasefield.phase_fracture_constitutive_model import PhaseFractureConstitutiveModel
class TestfracutreConstitutiveModel:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_fracture_constitutive_model(self, backend):
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 1)
        GD = mesh.geo_dimension()
#        tspace = TensorFunctionSpace(space, (-1, GD))
        tspace = TensorFunctionSpace(space, (GD, -1))

        uh = tspace.function()
        uh[2] = 1.0
        uh[-2] = 2.0

        E = 200
        nv = 0.3
        pfcm = PhaseFractureConstitutiveModel(E=E, nu=nv)
        strain = pfcm.strain(uh)
        stress0 = pfcm.effective_stress(strain=strain)
        stress1 = pfcm.effective_stress(u = uh)
        print(bm.max(bm.abs(stress0 - stress1)))
        assert bm.allclose(stress0, stress1)
        
        d = space.function()
        d[0] = 0.5
        stress = pfcm.stress(u = uh, d = d)
        
        NC = mesh.number_of_cells()
        NQ = int(pfcm.q*(pfcm.q+1)/2)
        H = bm.zeros((NC, NQ))
        H = pfcm.maximum_historical_strain_field(u=uh, H =H)
        print(H)




if __name__ == "__main__":
    TestfracutreConstitutiveModel().test_fracture_constitutive_model('numpy')
    TestfracutreConstitutiveModel().test_fracture_constitutive_model('pytorch')
    pytest.main(['test_fracture_constitutive_model.py', "-q"])   



