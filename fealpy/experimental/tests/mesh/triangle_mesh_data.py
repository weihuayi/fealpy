import numpy as np

# 定义多个典型的 TriangleMesh 对象
init_mesh_data = [
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

from_box_data= [
            {
                "node": np.array([[0 , 0], [0, 0.5], [0, 1], [0.5, 0], 
                    [0.5, 0.5], [0.5, 1], [1, 0], [1, 0.5], [1, 1]], dtype=np.float64),
                "edge": np.array([[1, 0], [0, 3], [4, 0], [2, 1], [1, 4], [5, 1], 
                    [5, 2], [3, 4], [3, 6], [7, 3], [4, 5], [4, 7], [8, 4], [8, 5], 
                    [6, 7], [7, 8]], dtype=np.int32), 
                "cell": np.array([[3, 4, 0], [6, 7, 3], [4, 5, 1], [7, 8, 4], 
                    [1, 0, 4], [4, 3, 7], [2, 1, 5], [5, 4, 8]], dtype=np.int32),
                "face2cell": np.array([[4, 4, 2, 2], [0, 0, 1, 1], [0, 4, 0, 0], [6, 6, 2, 2],
                    [2, 4, 1, 1], [2, 6, 0, 0], [6, 6, 1, 1], [0, 5, 2, 2], [1, 1, 1, 1], 
                    [1, 5, 0, 0], [2, 7, 2, 2], [3, 5, 1, 1], [3, 7, 0, 0], [7, 7, 1, 1], 
                    [1, 1, 2, 2], [3, 3, 2, 2]], dtype=np.int32),
                "NN": 9,
                "NE": 16,
                "NF": 16,
                "NC": 8
            }
]

entity_measure_data = [{"measure0": np.array([0.0], dtype=np.float64),
    "measure1": np.array([1.0, 1.0, np.sqrt(2)], dtype=np.float64),
    "measure": np.array([0.5], dtype=np.float64)}]
