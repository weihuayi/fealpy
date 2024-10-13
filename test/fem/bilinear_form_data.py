import numpy as np

mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        "cell": np.array([[0, 1, 2]], dtype=np.int64),
        "class": "TriangleMesh",
    },
    {
        "node": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        "cell": np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int64),
        "class": "TriangleMesh",
    }
]

