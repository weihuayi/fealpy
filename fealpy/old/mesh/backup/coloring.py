
import numpy as np 

def coloring(mesh, method='random', etype='node'):
    if method is 'random':
        c = randomcoloring(mesh)
    if method is 'random1':
        c = randomcoloring1(mesh)
    if method is 'random2':
        c = randomcoloring2(mesh)
    return c

def is_valid_coloring(mesh, c):
    edge = mesh.ds.edge
    if np.any(c[edge[:, 0]] == c[edge[:, 1]]):
        return False
    else:
        return True

def opt_coloring(mesh, c):
    mc = np.max(c)
    NN = mesh.number_of_nodes()

    edge = mesh.ds.edge

    nc = np.zeros((NN, mc), dtype=np.bool_)
    np.add.at(nc, (edge[:, 0], c[edge[:, 1]]-1), True)
    np.add.at(nc, (edge[:, 1], c[edge[:, 0]]-1), True)


def randomcoloring(mesh):
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()

    edge = mesh.ds.edge

    c = np.zeros(NN, dtype=np.int)

    isUnColor = (c == 0) 
    color = 0
    isRemainEdge = isUnColor[edge[:, 0]] & isUnColor[edge[:, 1]]
    while np.any(isRemainEdge):
        color += 1
        edge0 = edge[isRemainEdge]
        r = np.random.random(NN)
        isLess =  r[edge0[:, 0]] < r[edge0[:, 1]]
        flag = np.bincount(edge0[~isLess, 0], minlength=NN)
        flag += np.bincount(edge0[isLess, 1], minlength=NN)
        c[(flag == 0) & isUnColor] = color
        isUnColor = (c == 0) 

        isEdge0 = isUnColor[edge[:, 0]] & (c[edge[:, 1]] == color)
        isEdge1 = isUnColor[edge[:, 1]] & (c[edge[:, 0]] == color)
        flag = np.bincount(edge[isEdge0, 0], minlength=NN)
        flag += np.bincount(edge[isEdge1, 1], minlength=NN)
        isInteriorUnColor = (flag == 0) & isUnColor
        while np.any(isInteriorUnColor):
            isRemainEdge = isInteriorUnColor[edge[:,0]] & isInteriorUnColor[edge[:,1]]
            edge0 = edge[isRemainEdge]

            r = np.random.random(NN)
            isLess =  r[edge0[:, 0]] < r[edge0[:, 1]]
            flag = np.bincount(edge0[~isLess, 0], minlength=NN)
            flag += np.bincount(edge0[isLess, 1], minlength=NN)
            c[(flag == 0) & isInteriorUnColor] = color

            isUnColor = (c==0)
            isEdge0 = isUnColor[edge[:, 0]] & (c[edge[:, 1]] == color)
            isEdge1 = isUnColor[edge[:, 1]] & (c[edge[:, 0]] == color)
            flag = np.bincount(edge[isEdge0, 0], minlength=NN)
            flag += np.bincount(edge[isEdge1, 1], minlength=NN)
            isInteriorUnColor = (flag == 0) & isUnColor

        isRemainEdge = isUnColor[edge[:, 0]] & isUnColor[edge[:, 1]]

    if np.any(isUnColor):
        color += 1
        c[isUnColor] = color

    for i in range(1, color+1):
        print('There are %d points with color %d'%((c==i).sum(), i))

    return c

def randomcoloring1(mesh):
    NN = mesh.number_of_nodes()
    edge = mesh.ds.edge

    c = np.zeros(NN, dtype=np.int)

    isUnColor = (c == 0) 

    color = 1 # current color
    while np.any(isUnColor):
        r = np.random.random(NN)

        isEdge0 = isUnColor[edge[:, 0]]
        edge0 = edge[isEdge0, :]
        isLess = (~isUnColor[edge0[:, 1]]) | (r[edge0[:, 0]] < r[edge0[:, 1]])
        flag = np.bincount(edge0[~isLess, 0], minlength=NN)
        
        isEdge1 = isUnColor[edge[:, 1]]
        edge1 = edge[isEdge1, :]
        isLess =  (~isUnColor[edge1[:, 0]]) | (r[edge1[:, 1]] < r[edge1[:, 0]])
        flag += np.bincount(edge1[~isLess, 1], minlength=NN)
        isNewColor = isUnColor & (flag == 0)
        c[isNewColor] = color
        color += 1
        isUnColor = (c == 0) 
    return c

def randomcoloring2(mesh):
    N = mesh.number_of_points()
    NE = mesh.number_of_edges()

    edge = mesh.ds.edge

    c = np.zeros(N, dtype=np.int)

    isUnColor = (c == 0) 
    color = 1
    while NE > 0:
        isRemainEdge = isUnColor[edge[:, 0]] & isUnColor[edge[:, 1]]
        edge = edge[isRemainEdge, :]
        NE = len(edge)
        r = np.random.random(N)
        isLess =  r[edge[:, 0]] < r[edge[:, 1]]
        flag = np.bincount(edge[~isLess, 0], minlength=N)
        flag += np.bincount(edge[isLess, 1], minlength=N)
        c[(flag == 0) & isUnColor] = color
        
        color += 1
        isUnColor = (c == 0) 

    c[isUnColor] = color
    return c





    


