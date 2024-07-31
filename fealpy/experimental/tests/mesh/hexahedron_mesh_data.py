
import numpy as np

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax']

# 定义多个典型的 TriangleMesh 对象
mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [2, 0], [1, 2]], dtype=np.int32), 
        "cell": np.array([[0, 1, 2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 2, 2], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.int32),
        "NN": 3,
        "NE": 3,
        "NF": 3,
        "NC": 1
    },
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [2, 0], [3, 0], [1, 2], [2, 3]], dtype=np.int32),
        "cell": np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 2, 2], [0, 0, 2, 2],[1, 1, 1, 1]], dtype=np.int32),
        "NN": 4,
        "NE": 5, 
        "NF": 5,
        "NC": 2
    }
]
