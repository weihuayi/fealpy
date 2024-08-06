
import numpy as np

init_data= [
    {
        "ldof":3,
        "gdof":4,
        "interpolation":np.array([[0., 0.],
                                  [0., 1.],
                                  [1., 0.],
                                  [1., 1.]],dtype=np.int32),

        "c2d":np.array([[2, 3, 0],
                        [1, 0, 3]]),
        "f2d":np.array([[1, 0],
                        [0, 2],
                        [3, 0],
                        [3, 1],
                        [2, 3]]),
        "bdof":np.array([ True,  True,  True,  True]),

        "basis":np.array([[[0.2, 0.3, 0.5]],
                          [[0.1, 0.1, 0.8]]],dtype=np.float64),
        
        "gbasis":np.array([[[[ 1., -1.],
                             [ 0.,  1.],
                             [-1.,  0.]],

                            [[-1.,  1.],
                             [ 0., -1.],
                             [ 1.,  0.]]],


                          [[[ 1., -1.],
                            [ 0.,  1.],
                            [-1.,  0.]],

                           [[-1.,  1.],
                            [ 0., -1.],
                            [ 1.,  0.]]]],dtype=np.float64),
        "value":np.array([[2.3, 2.7],
                     [1.5, 3.5]]),

        "ltob":np.array([[1., 0.],
                         [0., 1.]]),

        "btol":np.array([[1., 0.],
                        [0., 1.]]),

    }
]
