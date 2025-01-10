from fealpy.backend import backend_manager as bm
from fealpy.cfd.sph.kdtree import Neighbor 
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pytest

class TestcKDTree:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_kdtree(self, backend):
        H = 0.4
        x = bm.arange(0, 1.2, 0.4)  
        y = bm.arange(0, 1.2, 0.4)  
        grid_x, grid_y = bm.meshgrid(x, y)
        points = bm.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)

        tree = cKDTree(points)

        def find_neighbors_within_distance(points, h):
            tree = cKDTree(points)  
            neighbors = tree.query_ball_tree(tree, h)
            return neighbors

        # 找到距离小于 2*H 的邻居
        idx = find_neighbors_within_distance(points, 2 * H)

        # 为每个点绘制图形
        for i, neighbors in enumerate(idx):
            plt.figure(figsize=(8, 6))
            plt.scatter(points[:, 0], points[:, 1], color='blue', label='All Points')
            plt.scatter(points[i, 0], points[i, 1], color='red', label='Current Point')  # 当前点

            # 绘制当前点和其邻居的连线
            for neighbor in neighbors:
                plt.plot([points[i, 0], points[neighbor, 0]], 
                         [points[i, 1], points[neighbor, 1]], 
                         color='gray', linestyle='--', alpha=0.5)

            plt.title(f'Point {i} and Its Neighbors')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.xlim(-0.1, 1.3)
            plt.ylim(-0.1, 1.3)
            plt.grid()
            plt.legend()
            plt.show()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_find_neighbors(self, backend):
        a = bm.array([0, 1, 3, 0, 1, 2, 4, 1, 2, 5, 0, 3, 4, 6, 
              1, 3, 4, 5, 7, 2, 4, 5, 8, 3, 6, 7, 
              4, 6, 7, 8, 5, 7, 8])
        b = bm.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 
                    3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 
                    6, 6, 7, 7, 7, 7, 8, 8, 8])
        
        H = 0.4
        x = bm.arange(0, 1.2, 0.4)  
        y = bm.arange(0, 1.2, 0.4)  
        grid_x, grid_y = bm.meshgrid(x, y)
        points = bm.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)
        nodedata = {
                    "position": points,
                }
        i_s, j_s = Neighbor.find_neighbors(nodedata, H)

        assert bm.all(bm.equal(i_s, a)), f"i_s does not match a: {i_s} != {a}"
        assert bm.all(bm.equal(j_s, b)), f"j_s does not match b: {j_s} != {b}"

if __name__ == "__main__":
    pytest.main()