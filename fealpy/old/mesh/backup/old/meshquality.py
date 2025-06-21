
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def show_mesh_quality(axes, quality):
    minq = np.min(quality)
    maxq = np.max(quality)
    meanq = np.mean(quality)
    hist, bins = np.histogram(quality, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    axes.bar(center, hist, align='center', width=0.02)
    axes.set_xlim(0, 1)

    #TODO: fix the textcoords warning
    axes.annotate('Min quality: {:.6}'.format(minq), xy=(0, 0), 
            xytext=(0.2, 0.8),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=20)
    axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0, 0),
            xytext=(0.2, 0.7),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=20)
    axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0, 0),
            xytext=(0.2, 0.6),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=20)
    return minq, maxq, meanq

class TriRadiusRatio():

    def __call__(self, point, cell):
        return self.quality(point, cell)

    def quality(self, point, cell):

        NC = cell.shape[0] 
        localEdge = np.array([(1, 2), (2, 0), (0, 1)])
        v = [point[cell[:,j],:] - point[cell[:,i],:] for i,j in localEdge]
        l = np.zeros((NC, 3))
        for i in range(3):
            l[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2 
        quality = p*q/(16*area**2)
        return quality

    def objective_function(self, point, cell):

        N = point.shape[0]
        NC = cell.shape[0]

        localEdge = np.array([(1, 2), (2, 0), (0, 1)])
        v = [point[cell[:,j],:] - point[cell[:,i],:] for i,j in localEdge]
        l = np.zeros((NC, 3))
        for i in range(3):
            l[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))

        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2 
        quality = p*q/(16*area**2)

        c = 1/l**2 + 1/(p.reshape(-1, 1)*l)
        b = np.zeros((NC, 6), dtype=np.float)
        ne = [1, 2, 0]
        pr = [2, 0, 1]
        W = np.array([[0, 1], [-1, 0]])
        weight = np.zeros(N, dtype=np.float)
        for i in range(3):
            ci = c[:, ne[i]] + c[:, pr[i]]
            np.add.at(weight, cell[:, i], quality*ci)
            b[:, [2*i, 2*i+1]] = ci.reshape(-1, 1)*point[cell[:, i]]
            b[:, [2*i, 2*i+1]] -= c[:, pr[i]].reshape(-1, 1)*point[cell[:, ne[i]]] 
            b[:, [2*i, 2*i+1]] -= c[:, ne[i]].reshape(-1, 1)*point[cell[:, pr[i]]]
            b[:, [2*i, 2*i+1]] -= (point[cell[:, pr[i]]] - point[cell[:, ne[i]]])@W/area.reshape(-1, 1)

        b *= quality.reshape(-1, 1) 

        gradF = np.zeros((N, 2), dtype=np.float)
        np.add.at(gradF[:, 0], cell.flatten(), b[:, [0, 2, 4]].flatten())
        np.add.at(gradF[:, 1], cell.flatten(), b[:, [1, 3, 5]].flatten())
        F = np.sum(quality)
        return F, gradF/weight.reshape(-1, 1)

    def is_valid(self, point, cell):
        v0 = point[cell[:, 2], :] - point[cell[:, 1], :]
        v1 = point[cell[:, 0], :] - point[cell[:, 2], :]
        v2 = point[cell[:, 1], :] - point[cell[:, 0], :]
        area = np.cross(v2, -v1)/2
        return np.all(area > 0)

    def show_quality(self, axes, q):
        return show_mesh_quality(axes, q)

class TetRadiusRatio:

    def __call__(self, point, cell):
        return self.quality(point, cell)

    def quality(self, point, cell):

