
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

    def __init__(self, alpha=2):
        self.alpha = alpha

    def __call__(self, mesh):
        return self.quality(mesh)

    def quality(self, mesh):
        point = mesh.point
        cell = mesh.ds.cell

        NC = mesh.number_of_cells()

        localEdge = mesh.ds.localEdge
        v = [point[cell[:,j],:] - point[cell[:,i],:] for i,j in localEdge]
        l = np.zeros((NC, 3))
        for i in range(3):
            l[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = mesh.area()
        quality = p*q/(16*area**2)
        return quality

    def objective_function(self, mesh):
        alpha = self.alpha

        point = mesh.point
        cell = mesh.ds.cell

        NC = mesh.number_of_cells()
        N = mesh.number_of_points()

        localEdge = mesh.ds.localEdge
        v = [point[cell[:,j],:] - point[cell[:,i],:] for i,j in localEdge]
        l = np.zeros((NC, 3))
        for i in range(3):
            l[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = mesh.area()
        quality = p*q/(16*area**2)
        mu= alpha*quality**alpha
        c = mu.reshape(-1, 1)*(1/l**2 + 1/(p.reshape(-1, 1)*l))

        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))

        I = np.concatenate((
            cell[:, 0], cell[:, 0], cell[:, 0],
            cell[:, 1], cell[:, 1], cell[:, 1],
            cell[:, 2], cell[:, 2], cell[:, 2]))
        J = np.concatenate((
            cell[:, 0], cell[:, 1], cell[:, 2],
            cell[:, 0], cell[:, 1], cell[:, 2],
            cell[:, 0], cell[:, 1], cell[:, 2],
            ))
        A = csr_matrix((val, (I, J)), shape=(N, N), dtype=np.float)

        cn = mu/area
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((cell[:, 0], cell[:, 0], cell[:, 1], cell[:, 1], cell[:, 2], cell[:, 2]))
        J = np.concatenate((cell[:, 1], cell[:, 2], cell[:, 0], cell[:, 2], cell[:, 0], cell[:, 1]))
        B = csr_matrix((val, (I, J)), shape=(N, N), dtype=np.float)

        F = np.sum(quality**alpha)
        gradF = np.zeros((N, 2), dtype=np.float)
        gradF[:, 0] = A@point[:, 0] + B@point[:, 1]
        gradF[:, 1] = point[:, 0]@B + A@point[:, 1]
        return F, gradF, A, B 

    def show_quality(self, axes, q):
        return show_mesh_quality(axes, q)
