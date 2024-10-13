
import numpy as np


init_mesh_data = [
    {
        "node": np.array([[0.] , [1.]], dtype=np.float64),
        "edge": np.array([[0,1]], dtype=np.int32), 
        "face": np.array([[0] , [1]] , dtype = np.int32),
        "cell": np.array([[0,1]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 0 ,0], [0, 0, 1, 1]], dtype=np.int32),
        "cell2face": np.array([[0, 1]] , dtype = np.int32),
        "cell2edge": np.array([[0]] , dtype = np.int32),
        "NN": 2,
        "NE": 1,
        "NF": 2,
        "NC": 1
    },
    {
        "node": np.array([[0.], [0.5] , [1.]], dtype=np.float64),
        "edge": np.array([[0, 1], [1,2]], dtype=np.int32),
        "face": np.array([[0],[1],[2]] , dtype=np.int32),
        "cell": np.array([[0,1] , [1,2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 0,0],[0,1,1,0],[1,1,1,1]], dtype=np.int32),
        "cell2face": np.array([[0,1] , [1,2]] , dtype = np.int32),
        "cell2edge": np.array([[0],[1]],dtype = np.int32),
        "NN": 3,
        "NE": 2, 
        "NF": 3,
        "NC": 2
    }
]

from_interval_domain_data = [
    {
        "interval" : np.array([0,1] , dtype = np.float64),
        "n" : 2,
        "node": np.array([[0.], [0.5] , [1.]], dtype=np.float64),
        "edge": np.array([[0, 1], [1,2]], dtype=np.int32),
        "face": np.array([[0],[1],[2]] , dtype=np.int32),
        "cell": np.array([[0,1] , [1,2]], dtype=np.int32),
        "face2cell": np.array([[0, 0, 0,0],[0,1,1,0],[1,1,1,1]], dtype=np.int32),
        "cell2face": np.array([[0,1] , [1,2]] , dtype = np.int32),
        "cell2edge": np.array([[0],[1]],dtype = np.int32),
        "NN": 3,
        "NE": 2, 
        "NF": 3,
        "NC": 2
        
    }
]

from_mesh_boundary_data = [
    {
        "node": np.array([[0.       , 0.        ],
                          [0.       , 0.33333333],
                          [0.       , 0.66666667],
                          [0.       , 1.        ],
                          [0.33333333 ,0.        ],
                          [0.33333333 ,1.        ],
                          [0.66666667 ,0.        ],
                          [0.66666667 ,1.        ],
                          [1.         ,0.        ],
                          [1.         ,0.33333333],
                          [1.         ,0.66666667],
                          [1.         ,1.        ]] , np.float64),

        "edge": np.array([[ 1 , 0],
                        [ 0 , 4],
                        [ 2 , 1],
                        [ 3 , 2],
                        [ 5 , 3],
                        [ 4 , 6],
                        [ 7 , 5],
                        [ 6 , 8],
                        [11 , 7],
                        [ 8 , 9],
                        [ 9 ,10],
                        [10 ,11]],np.int32),
        
        "face": np.array([[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6],[ 7],[ 8],[ 9],[10],[11]] , np.int32),
        
        "cell": np.array([[ 1 , 0],
                        [ 0 , 4],
                        [ 2 , 1],
                        [ 3 , 2],
                        [ 5 , 3],
                        [ 4 , 6],
                        [ 7 , 5],
                        [ 6 , 8],
                        [11 , 7],
                        [ 8 , 9],
                        [ 9 ,10],
                        [10 ,11]],np.int32),
        
        "face2cell": np.array( [[ 0 , 1 , 1 , 0],
                                [ 0 , 2 , 0 , 1],
                                [ 2 , 3 , 0 , 1],
                                [ 3 , 4 , 0 , 1],
                                [ 1 , 5 , 1 , 0],
                                [ 4 , 6 , 0 , 1],
                                [ 5 , 7 , 1 , 0],
                                [ 6 , 8 , 0 , 1],
                                [ 7 , 9 , 1 , 0],
                                [ 9 ,10 , 1 , 0],
                                [10 ,11 , 1 , 0],
                                [ 8 ,11 , 0 , 1]], np.int32),
        
        "cell2face": np.array( [[ 1 , 0],
                                [ 0 , 4],
                                [ 2 , 1],
                                [ 3 , 2],
                                [ 5 , 3],
                                [ 4 , 6],
                                [ 7 , 5],
                                [ 6 , 8],
                                [11 , 7],
                                [ 8 , 9],
                                [ 9 ,10],
                                [10 ,11]] , np.int32),

        "cell2edge": np.array( [[ 0],[ 1],[ 2],[ 3],[ 4],[ 5],[ 6],[ 7],[ 8],[ 9],[10],[11]]),

        "NN": 12,
        "NE": 12, 
        "NF": 12,
        "NC": 12
        
    }
]
from_circle_boundary_data = [
    {
        "center" : np.array([0.0,0.0] , dtype = np.float64),
        "radius" : 1,
        "n" : 10,
        "node": np.array([[ 1.00000000e+00 , 0.00000000e+00],
                          [ 8.09016994e-01 , 5.87785252e-01],
                          [ 3.09016994e-01 , 9.51056516e-01],
                          [-3.09016994e-01 , 9.51056516e-01],
                          [-8.09016994e-01 , 5.87785252e-01],
                          [-1.00000000e+00 , 1.22464680e-16],
                          [-8.09016994e-01 ,-5.87785252e-01],
                          [-3.09016994e-01 ,-9.51056516e-01],
                          [ 3.09016994e-01 ,-9.51056516e-01],
                          [ 8.09016994e-01 ,-5.87785252e-01]] , dtype = np.float64),
        "edge": np.array([[0 ,1],
                          [1 ,2],
                          [2 ,3],
                          [3 ,4],
                          [4 ,5],
                          [5 ,6],
                          [6 ,7],
                          [7 ,8],
                          [8 ,9],
                          [9 ,0]] , dtype = np.int32),
        "face": np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]], dtype = np.int32),
        "cell": np.array( [[0, 1],
                           [1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5],
                           [5, 6],
                           [6, 7],
                           [7, 8],
                           [8, 9],
                           [9, 0]] , dtype = np.int32),
        "face2cell": np.array( [[0, 9, 0, 1],
                                 [0, 1, 1, 0],
                                 [1, 2, 1, 0],
                                 [2, 3, 1, 0],
                                 [3, 4, 1, 0],
                                 [4, 5, 1, 0],
                                 [5, 6, 1, 0],
                                 [6, 7, 1, 0],
                                 [7, 8, 1, 0],
                                 [8, 9, 1, 0]] , dtype = np.int32),
        "cell2face": np.array( [[0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 4],
                                [4, 5],
                                [5, 6],
                                [6, 7],
                                [7, 8],
                                [8, 9],
                                [9, 0]] , dtype = np.int32),
        "cell2edge": np.array( [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]] , dtype = np.int32),
        "NN":10,
        "NE":10,
        "NF":10,
        "NC":10

    }
]

integrator_data = [
    {
        "q": 2,
        "bcs": np.array([[0.78867513 , 0.21132487],
                         [0.21132487 , 0.78867513]] , np.float64),
        "ws": np.array([0.5 ,0.5], np.float64)
    },
    {  
        "q": 4,
        "bcs": np.array([[0.93056816, 0.06943184],
                         [0.66999052, 0.33000948],
                         [0.33000948, 0.66999052],
                         [0.06943184, 0.93056816]] , np.float64),
        "ws": np.array([0.17392742,0.32607258, 0.32607258 ,0.17392742], np.float64)
    }
]

grad_shape_function_data = [
    {
        "p" : 2,
        "gphi" : np.array([[[[-2.15470054],
         [ 2.30940108],
         [-0.15470054]],

        [[-2.15470054],
         [ 2.30940108],
         [-0.15470054]],

        [[-2.15470054],
         [ 2.30940108],
         [-0.15470054]]],


       [[[ 0.15470054],
         [-2.30940108],
         [ 2.15470054]],

        [[ 0.15470054],
         [-2.30940108],
         [ 2.15470054]],

        [[ 0.15470054],
         [-2.30940108],
         [ 2.15470054]]]] , dtype = np.float64),

    },
    {
        "p": 3,
        "gphi" : np.array([[[[-2.29903811],
         [ 1.29903811],
         [ 1.29903811],
         [-0.29903811]],

        [[-2.29903811],
         [ 1.29903811],
         [ 1.29903811],
         [-0.29903811]],

        [[-2.29903811],
         [ 1.29903811],
         [ 1.29903811],
         [-0.29903811]]],


       [[[ 0.29903811],
         [-1.29903811],
         [-1.29903811],
         [ 2.29903811]],

        [[ 0.29903811],
         [-1.29903811],
         [-1.29903811],
         [ 2.29903811]],

        [[ 0.29903811],
         [-1.29903811],
         [-1.29903811],
         [ 2.29903811]]]] , dtype = np.float64)
    }
]

entity_measure_data = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "cm" : np.array([1., 1., 1.] , dtype= np.float64),
        "nm" : np.array([0.] , dtype = np.float64),
        "em" : np.array([1., 1., 1.] , dtype= np.float64),
        "fm" : np.array([0.] , dtype = np.float64)
    },
    {
        "node" : np.array([0,0.5,1,2],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "cm" : np.array([0.5, 0.5, 1. ] , dtype= np.float64),
        "nm" : np.array([0.] , dtype= np.float64),
        "em" : np.array([0.5, 0.5, 1. ],dtype= np.float64),
        "fm" : np.array([0.],dtype= np.float64)
    }
]

grad_lambda_data = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "Dlambda" : np.array([[[-1.],
                               [ 1.]],
                          
                              [[-1.],
                               [ 1.]],
                          
                              [[-1.],
                               [ 1.]]],dtype = np.float64)
    }
]

prolongation_matrix_data = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "p0" : 1,
        "p1" : 2,
        "prolongation_matrix" : np.array([[1. , 0. , 0. , 0. ],
                                          [0. , 1. , 0. , 0. ],
                                          [0. , 0. , 1. , 0. ],
                                          [0. , 0. , 0. , 1. ],
                                          [0.5, 0.5, 0. , 0. ],
                                          [0. , 0.5, 0.5, 0. ],
                                          [0. , 0. , 0.5, 0.5]] , dtype= np.float64)
    }
]

number_of_local_ipoints_data = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "p" : 4,
        "nlip" : 5
    }
]

interpolation_points_data  = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "p" : 1,
        "ipoints" : np.array([[0.],[1.],[2.],[3.]] , dtype = np.float64 )
    },
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "p" : 4,
        "ipoints" : np.array([[0.  ],[1.  ],[2.  ],[3.  ],[0.25],[0.5 ],
                             [0.75],[1.25],[1.5 ],[1.75],[2.25],[2.5 ],[2.75]], np.float64)
    }
]

cell_normal_data = [
    {
        "center" : np.array([0.0,0.0] , dtype = np.float64),
        "radius" : 1.0,
        "n" : 10,
        "cn" : np.array([[ 5.87785252e-01,  1.90983006e-01],
                         [ 3.63271264e-01,  5.00000000e-01],
                         [ 1.11022302e-16,  6.18033989e-01],
                         [-3.63271264e-01,  5.00000000e-01],
                         [-5.87785252e-01,  1.90983006e-01],
                         [-5.87785252e-01, -1.90983006e-01],
                         [-3.63271264e-01, -5.00000000e-01],
                         [-1.11022302e-16, -6.18033989e-01],
                         [ 3.63271264e-01, -5.00000000e-01],
                         [ 5.87785252e-01, -1.90983006e-01]],dtype = np.float64)
    }
]

uniform_refine_data = [
    {
        "node_init" : np.array([[0.] , [1.]], dtype=np.float64),
        "cell_init" : np.array([[0,1]], dtype=np.int32),
        "n" : 2,
        "node" : np.array([[0.  ],
                           [1.  ],
                           [0.5 ],
                           [0.25],
                           [0.75]], dtype= np.float64),
        "edge" : np.array([[0, 3],
                           [2, 4],
                           [3, 2],
                           [4, 1]], dtype=np.int32),
        "face" : np.array([[0],[1],[2],[3],[4]], dtype=np.int32),
        "cell" : np.array([[0, 3],
                           [2, 4],
                           [3, 2],
                           [4, 1]], dtype=np.int32),
        "face2cell" : np.array([[0, 0, 0, 0],
                               [3, 3, 1, 1],
                               [1, 2, 0, 1],
                               [0, 2, 1, 0],
                               [1, 3, 1, 0]], dtype=np.int32),
        "cell2face" : np.array([[0, 3],
                                [2, 4],
                                [3, 2],
                                [4, 1]], dtype=np.int32),
        "cell2edge" : np.array([[0],[1],[2],[3]], dtype=np.int32),
        "edge2cell" : np.array([[0],[1],[2],[3]], dtype=np.int32),
        "NN": 5,
        "NE": 4, 
        "NF": 5,
        "NC": 4
    }
]
refine_data  = [
    {
        "node_init" : np.array([[0.] ,[0.5] ,  [1.]], dtype=np.float64),
        "cell_init" : np.array([[0,1], [1,2]], dtype=np.int32),
        "isMarkedCell" : np.array([True,False]),
        "node" : np.array([[0.] ,[0.5] ,  [1.] ,[0.25]], dtype=np.float64),
        "edge" : np.array([[0,3],[1,2],[3,1]],dtype=np.int32),
        "face" : np.array([[0],[1],[2],[3]],dtype = np.int32),
        "cell" : np.array([[0,3],[1,2],[3,1]],dtype=np.int32),
        "face2cell" : np.array([[0, 0, 0, 0],
                                [1, 2, 0, 1],
                                [1, 1, 1, 1],
                                [0, 2, 1, 0]] , dtype = np.int32),
        "cell2face" : np.array([[0,3],[1,2],[3,1]],dtype=np.int32),
        "cell2edge" : np.array([[0],[1],[2]],dtype=np.int32),
        "edge2cell" : np.array([[0],[1],[2]],dtype=np.int32),
        "NN": 4,
        "NE": 3,
        "NF": 4,
        "NC": 3
    }
]
quadrature_formula_data = [
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "q"  : 1,
        "qf1" : np.array([0.5, 0.5]),
        "qf2" : 1.0
    },
    {
        "node" : np.array([0,1,2,3],dtype=np.float64),
        "cell" : np.array([[0,1],[1,2],[2,3]],dtype=np.int32),
        "q"  : 3,
        "qf1" : np.array([0.88729833, 0.11270167]),
        "qf2" : 0.2777777777777778
    }
]
