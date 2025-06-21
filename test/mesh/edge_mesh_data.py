import numpy as np

# 定义多个典型的 TriangleMesh 对象

init_mesh_data = [
    {
        "node": np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
        "edge": np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int32), 
        "cell": np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int32),
        "cell2node": np.array([[0, 1],
                               [0, 2],
                               [1, 2],
                               [1, 3]]),
        "NN": 4,
        "NE": 4,
        "NC": 4,
        
        
        "edge_tangent":np.array([[ 1.,  0.],
                                 [ 0.,  1.],
                                 [-1.,  1.],
                                 [ 0.,  1.]]),
        
        "edge_length":np.array([1.        , 1.        , 1.41421356, 1.        ],dtype = np.float64),
        
        
        "grad_lambda":np.array([[[-1. , -0. ],
                                 [ 1. ,  0. ]],

                                [[-0. , -1. ],
                                 [ 0. ,  1. ]],

                                [[ 0.5, -0.5],
                                 [-0.5,  0.5]],

                                [[-0. , -1. ],
                                 [ 0. ,  1. ]]]),
 
        "interpolation_points":np.array([[0. , 0. ],
                                         [1. , 0. ],
                                         [0. , 1. ],
                                         [1. , 1. ],
                                         [0.5, 0. ],
                                         [0. , 0.5],
                                         [0.5, 0.5],
                                         [1. , 0.5]]) , # p=2
        
        "cell_normal":np.array([[ 0., -1.],
                                [ 1.,  0.],
                                [ 1.,  1.],
                                [ 1.,  0.]])

        
    }
]






