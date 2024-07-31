import numpy as np

# Define multiple typical UniformMesh objects
mesh_data = [
    {
        "extent": (0, 2, 0, 2),
        "h": (1.0, 1.0),
        "origin": (0, 0),
        "node": np.array([[[0, 0], [0, 1], [0, 2]],
                          [[1, 0], [1, 1], [1, 2]],
                          [[2, 0], [2, 1], [2, 2]]], dtype=np.float64),
        "edge": np.array([[0, 3],
                [1, 4],
                [5, 2],
                [3, 6],
                [4, 7],
                [8, 5],
                [1, 0],
                [2, 1],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8]], dtype=np.int32),
        "face": np.array([[0, 3],
                          [1, 4],
                          [5, 2],
                          [3, 6],
                          [4, 7],
                          [8, 5],
                          [1, 0],
                          [2, 1],
                          [3, 4],
                          [4, 5],
                          [6, 7],
                          [7, 8]], dtype=np.int32),

        "cell": np.array([[0, 1, 3, 4],
 [1, 2, 4, 5],
 [3, 4, 6, 7],
 [4, 5, 7, 8]], dtype=np.int32
),

        "NN": 9,
        "NE": 12,
        "NF": 12,
        "NC": 4,

        "edge_length": (1.0, 1.0),
        "cell_area": 1.0
    }]