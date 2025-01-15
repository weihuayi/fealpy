from fealpy.backend import backend_manager as bm
from fealpy.cfd.sph.kdtree import Neighbor 
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pytest

class TestcKDTree:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_find_neighbors(self, backend):
        bm.set_backend(backend)
        a = bm.array([0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,
            3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8])
        b = bm.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,
            4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8])
        
        H = 0.4
        x = bm.arange(0, 1.2, 0.4)  
        y = bm.arange(0, 1.2, 0.4)  
        grid_x, grid_y = bm.meshgrid(x, y, indexing='ij')
        points = bm.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)
        nodedata = {
                    "position": points,
                }
        i_s, j_s = Neighbor.find_neighbors(nodedata, H)
        print("i_s 的类型:", type(i_s))
        print("j_s 的类型:", type(j_s))
        assert bm.all(bm.equal(i_s, a)), f"i_s does not match a: {i_s} != {a}"
        assert bm.all(bm.equal(j_s, b)), f"j_s does not match b: {j_s} != {b}"

if __name__ == "__main__":
    pytest.main()