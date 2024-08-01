
import numpy as np



init_data  = [
    {
        "node": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3)/2, 0.0],
                [0.5, np.sqrt(3)/6, np.sqrt(2/3)]], dtype=np.float64),
        "cell": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "edge": np.array([[0, 1],
               [0, 2],
               [0, 3],
               [1, 2],
               [1, 3],
               [2, 3]]),
        "face": np.array([[0, 2, 1],
               [0, 1, 3],
               [0, 3, 2],
               [1, 2, 3]]),
        "face2cell": np.array([[0, 0, 3, 3],
               [0, 0, 2, 2],
               [0, 0, 1, 1],
               [0, 0, 0, 0]]),
        "NN": 4,
        "NE": 6,
        "NF": 4,
        "NC": 1
    },
]

from_one_tetrahedron_data  = [
    {
        "meshtype": 'equ',
        "edge": np.array([[0, 1],
               [0, 2],
               [0, 3],
               [1, 2],
               [1, 3],
               [2, 3]]),
        "face": np.array([[0, 2, 1],
               [0, 1, 3],
               [0, 3, 2],
               [1, 2, 3]]),
        "face2cell": np.array([[0, 0, 3, 3],
               [0, 0, 2, 2],
               [0, 0, 1, 1],
               [0, 0, 0, 0]]),
        "NN": 4,
        "NE": 6,
        "NF": 4,
        "NC": 1
    },
    {
        "meshtype": 'iso',
        "edge": np.array([[0, 1],
               [0, 2],
               [0, 3],
               [1, 2],
               [1, 3],
               [2, 3]]),
        "face": np.array([[0, 2, 1],
               [0, 1, 3],
               [0, 3, 2],
               [1, 2, 3]]),
        "face2cell": np.array([[0, 0, 3, 3],
               [0, 0, 2, 2],
               [0, 0, 1, 1],
               [0, 0, 0, 0]]),
        "NN": 4,
        "NE": 6,
        "NF": 4,
        "NC": 1
    },
]
