import pytest

from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager
from model_init_data import init_model_data


class TestPDEModelManager:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("modeldata", init_model_data)
    def test_model_init(self, modeldata, backend):
        bm.set_backend(backend)
        pde_name = modeldata['PDE']
        example_name = modeldata['example_name']
        pde = PDEModelManager(pde_name).get_example(example_name)
        print(pde.__doc__)
        mesh = pde.init_mesh()
        NN = mesh.number_of_nodes()



if __name__ == "__main__":
    pytest.main(["./test_model_init.py", "-k", "test_model_init"])