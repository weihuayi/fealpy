import numpy as np

init_mesh_data = [
    {
        "extent": (0, 2),
        "h": 1.0,
        "origin": 0,

        # "node": np.array([[[0, 0], [0, 1], [0, 2]],
        #                   [[1, 0], [1, 1], [1, 2]],
        #                   [[2, 0], [2, 1], [2, 2]]], dtype=np.float64),
        "node": np.array([[0], [1], [2]], dtype=np.float64),

        "edge": np.array([[0, 1],
                          [1, 2]], dtype=np.int32),

        "face": np.array([[0], [1], [2]], dtype=np.float64),

        "cell": np.array([[0, 1],
                          [1, 2]], dtype=np.int32),

        "NN": 3,
        "NE": 2,
        "NF": 3,
        "NC": 2,
    }
]

entity_measure_data = [
    {
        "extent": (0, 1),
        "h": 1.0,
        "origin": 0,

    }
]

interpolation_points_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,

        "ipoints_p1": np.array([[0.],
                                [0.1],
                                [0.2],
                                [0.3],
                                [0.4],
                                [0.5],
                                [0.6],
                                [0.7],
                                [0.8],
                                [0.9],
                                [1.]], dtype=np.float64),
        "ipoints_p2": np.array([[0.],
                                [0.1],
                                [0.2],
                                [0.3],
                                [0.4],
                                [0.5],
                                [0.6],
                                [0.7],
                                [0.8],
                                [0.9],
                                [1.],
                                [0.05],
                                [0.15],
                                [0.25],
                                [0.35],
                                [0.45],
                                [0.55],
                                [0.65],
                                [0.75],
                                [0.85],
                                [0.95]], dtype=np.float64),
    }
]


entity_to_ipoints_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,

    }
]

quadrature_formula_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,

        "bcs_q1": (np.array([[0.5, 0.5]], dtype=np.float64)),
        "bcs_q2": (np.array([[0.78867513, 0.21132487],
                             [0.21132487, 0.78867513]], dtype=np.float64)),
    }
]

barycenter_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,

        "cell_barycenter": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float64),
        
        "edge_barycenter": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float64),

    }
]

uniform_refine_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,
        "node_refined": np.array([
            [0.00], [0.05], [0.10], [0.15],
            [0.20], [0.25], [0.30], [0.35],
            [0.40], [0.45], [0.50], [0.55],
            [0.60], [0.65], [0.70], [0.75],
            [0.80], [0.85], [0.90], [0.95], [1.]
        ]),
    }
]

boundary_data = [
    {
        "extent": (0, 10),
        "h": 0.1,
        "origin": 0,

        "boundary_node_flag": np.array([
    True, False, False, False, False, False, False, False, False, False, True
], dtype=bool),

        "boundary_edge_flag": np.array([
            True, False, False, False, False, False, False, False, False, True
        ], dtype=bool),

        "boundary_face_flag": np.array([
            True, False, False, False, False, False, False, False, False, False, True
        ], dtype=bool),

        "boundary_cell_flag": np.array([
            True, False, False, False, False, False, False, False, False, True
        ], dtype=bool),
    }
]