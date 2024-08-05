import numpy as np

triangle_mesh_one_box = [
        {
            "number_of_local_dofs": 6,
            "number_of_global_dofs": 9,
            "interpolation points":np.array([
                                 [[0., 0.]],
                                 [[0., 1.]],
                                 [[1., 0.]],
                                 [[1., 1.]],
                                 [[0. , 0.5]],
                                 [[0.5, 0. ]],
                                 [[0.5, 0.5]],
                                 [[0.5, 1. ]],
                                 [[1. , 0.5]],], dtype=np.float64), 
            "cell_to_dof":np.array([
                                 [[2, 8, 5, 3, 6, 0]],
                                 [[1, 4, 7, 0, 6, 3]],], dtype=np.int32),
            "face_to_dof":np.array([
                                 [[1, 4, 0]],
                                 [[0, 5, 2]],
                                 [[3, 6, 0]],
                                 [[3, 7, 1]],
                                 [[2, 8, 3]],
                                ]),
            "edge_to_dof":np.array([
                                 [[1, 4, 0]],
                                 [[0, 5, 2]],
                                 [[3, 6, 0]],
                                 [[3, 7, 1]],
                                 [[2, 8, 3]],
                                ])

        }



        ]
