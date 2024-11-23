import ipdb

import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from app.fracturex.fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from app.fracturex.fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory

from app.fracturex.fracturex.phasefield.main_solve import MainSolve

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
        material_properties = {'E': E, 'nu': nv, 'Gc': 1.0, 'l': 0.1}
        gd = EDFunc()
        pfcm = PhaseFractureMaterialFactory.create('HybridModel', material_properties, gd)
        
        qf = mesh.integrator(q=3, etype='cell')
        bc, ws = qf.get_quadrature_points_and_weights()
        
        pfcm.update_disp(uh)
        tuh = pfcm._uh
        print('uh', bm.max(bm.abs(tuh - uh)))

        uh[2] = 0.5
        pfcm.update_disp(uh)
        tuh = pfcm._uh  
        print('uh', bm.max(bm.abs(tuh - uh)))

        strain = pfcm.strain_value(bc)
        stress1 = pfcm.effective_stress(bc = bc)
       


        d = space.function()

        d[0] = 0.5
        pfcm.update_phase(d)
        
        stress = pfcm.stress_value(bc)
        matrix = pfcm.elastic_matrix(bc)
        print('stress', stress.shape)
        print('matrix', matrix.shape)
        
        NC = mesh.number_of_cells()

        H = pfcm.maximum_historical_field(bc)
        print('H', H.shape,H)
        





if __name__ == "__main__":
    TestfracutreConstitutiveModel().test_fracture_constitutive_model('numpy')
    TestfracutreConstitutiveModel().test_fracture_constitutive_model('pytorch')
    pytest.main(['test_fracture_constitutive_model.py', "-q"])   



