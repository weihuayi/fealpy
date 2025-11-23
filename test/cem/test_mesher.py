from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from fealpy.cem.mesh import MetalensesMesher

import pytest


class TestMetalensesMesher:
    @pytest.fixture
    def mesher(self):
        # Example parameters, replace with actual values as needed
        metalenses_params = {
            "glass_size": 800,
            "glass_height": 3000,
            "air_layer_height": 4800,
            "bottom_pml_height": 960,
            "top_pml_height": 960,
            "antenna1_size": 190,
            "antenna1_height": 600,
            "antenna2_size": 160,
            "antenna2_height": 600,
            "antenna3_size": 160,
            "antenna3_height": 600,
            "antenna4_size": 160,
            "antenna4_height": 600
        }
        return MetalensesMesher(metalenses_params)

    def test_generate_mesh(self, mesher):
        unit_mesh = mesher.generate(mesh_size=0.2)
        # unit_mesh.to_vtk(fname='unit_metalenses_tet_mesh.vtu')
        assert isinstance(unit_mesh, TetrahedronMesh)
        assert unit_mesh.number_of_cells() > 0
