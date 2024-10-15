import numpy as np

def mark(eta, theta, method='L2'):
    isMarked = np.zeros(len(eta), dtype=np.bool_)
    if method is 'MAX':
        isMarked[eta > theta*np.max(eta)] = True
    elif method is 'COARSEN':
        isMarked[eta < theta*np.max(eta)] = True
    elif method is 'L2':
        eta = eta**2
        idx = np.argsort(eta)[-1::-1]
        x = np.cumsum(eta[idx])
        isMarked[idx[x < theta*x[-1]]] = True
        isMarked[idx[0]] = True
    else:
        raise ValueError("I have not code the method")
    markedCell, = np.nonzero(isMarked)
    return markedCell

class AdaptiveMarker():
    def __init__(self, eta, theta=0.2, ctheta=0.1):
        self.eta = eta
        self.theta = theta
        self.ctheta = ctheta

    def refine_marker(self, tree):
        leafCellIdx = tree.leaf_cell_index()
        NC = tree.number_of_cells()
        eta = np.zeros(NC, dtype=tree.ftype)
        idxmap = tree.celldata['idxmap']
        np.add.at(eta, idxmap, self.eta)
        markedIdx = mark(eta[leafCellIdx], self.theta)
        print(eta[leafCellIdx])
        return leafCellIdx[markedIdx]

    def coarsen_marker(self, qtmesh):
        idx = qtmesh.leaf_cell_index()
        markedIdx = mark(self.eta, self.ctheta, method='COARSEN')
        return idx[markedIdx]
