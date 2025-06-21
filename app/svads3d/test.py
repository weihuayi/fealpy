import numpy as np

def compute_ground_coordinates():
    location = np.array([[ 4.175, -1.735],
                         [-4.175, -1.735],
                         [-8.75 ,  0.0  ],
                         [-4.175,  1.735],
                         [ 4.175,  1.735],
                         [ 8.75 ,  0.0  ]])

    v = np.array([[0, -1], [0, -1], [-1, 0], [0, 1], [0, 1], [1, 0]])

    rot = np.array([[[-1, 0], [0, -1]], 
                    [[-1, 0], [0, -1]],
                    [[ 0, -1], [1, 0]],
                    [[ 1, 0], [0, 1]],
                    [[ 1, 0], [0, 1]],
                    [[0, 1], [-1, 0]]])

    x = np.linspace(-0.25, 0.25, 6, endpoint=True)
    y = np.linspace(0.4, 0.1, 4, endpoint=True)
    x, y = np.meshgrid(x, y)
    point_loc = np.zeros([24, 2], dtype=np.float64)
    point_loc[:, 0] = x.flatten()
    point_loc[:, 1] = y.flatten()

    ground_points = [location[i] + v[i] + np.dot(rot[i], point_loc.T).T for i in range(6)]
    print(ground_points)


    return point_loc 

compute_ground_coordinates()
