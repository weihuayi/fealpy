import numpy as np

# 定义多个典型的 TriangleMesh 对象

init_mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int32), 
        "cell": np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int32),
        "face2cell": np.array([[ True,  True, False, False],[ True, False,  True,  True],[False,  True,  True, False],[False, False, False,  True]],dtype = np.bool_),
        "NN": 4,
        "NE": 4,
        "NC": 4
    },
]






