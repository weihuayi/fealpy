import numpy as np 
from numpy.linalg import norm, det, inv


class QualityMetric:
    def show(self, q):
        print("\n质量最大值:\t", max(q))
        print("质量最小值:\t", min(q))
        print("质量平均值:\t", np.mean(q))
        print("质量均方根:\t", np.sqrt(np.mean(q**2)))
        print("质量标准差:\t", np.std(q))
    def show_mesh_quality(self,axes, quality):
        minq = np.min(quality)
        maxq = np.max(quality)
        meanq = np.mean(quality)
        rmsq = np.sqrt(np.mean(quality**2))
        stdq = np.std(quality)
        hist, bins = np.histogram(quality, bins=50, range=(0, 1))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=0.02)
        axes.set_xlim(0, 1)

        #TODO: fix the textcoords warning
        axes.annotate('Min quality: {:.6}'.format(minq), xy=(0, 0), 
                xytext=(0.15, 0.85),
                textcoords="figure fraction",
                horizontalalignment='left', verticalalignment='top', fontsize=15)
        axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0, 0),
                xytext=(0.15, 0.8),
                textcoords="figure fraction",
                horizontalalignment='left', verticalalignment='top', fontsize=15)
        axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0, 0),
                xytext=(0.15, 0.75),
                textcoords="figure fraction",
                horizontalalignment='left', verticalalignment='top', fontsize=15)
        axes.annotate('RMS: {:.6}'.format(rmsq), xy=(0, 0),
                xytext=(0.15, 0.7),
                textcoords="figure fraction",
                horizontalalignment='left', verticalalignment='top', fontsize=15)
        axes.annotate('STD: {:.6}'.format(stdq), xy=(0, 0),
                xytext=(0.15, 0.65),
                textcoords="figure fraction",
                horizontalalignment='left', verticalalignment='top', fontsize=15)
                
        return minq, maxq, meanq, rmsq, stdq


class InverseMeanRatio(QualityMetric):
    def __init__(self, w):
        self.invw = inv(w)

    def quality(self, mesh):
        """
        @brief 计算每个网格单元顶点的质量
        """
        J = mesh.jacobian_matrix()
        T = J@self.invw 
        q = 0.5*norm(T, axis=(-2, -1))/det(T)
        return q


class TriRadiusRatio(QualityMetric):
    def quality(self, mesh, return_grad=False):
        """
        @brief 计算半径比质量

        mu = R/(2*r)
        p = l_0 + l_1 + l_2
        q = l_0 * l_1 * l_2
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()

        localEdge = mesh.ds.local_edge()
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
        J = np.zeros((NC,2,2))
        J[:,0]=v[2]
        J[:,1]=-v[1]
        detJ = np.linalg.det(J)
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[1], v[2])/2
        quality = (p*q)/(16*area**2)
        '''
        if return_grad:
            grad = np.zeros((NC, 3, GD), dtype=mesh.ftype)
            grad[:, 0, :]  = (1/p/l[:, 1] + 1/l2[:, 1])[:, None]*(node[cell[:, 0]] - node[cell[:, 2]])
            grad[:, 0, :] += (1/p/l[:, 2] + 1/l2[:, 2])[:, None]*(node[cell[:, 0]] - node[cell[:, 1]])
            grad[:, 0, :] += 
        
        else:
            return quality
        '''
        quality[detJ<0]=0
        return quality
    def grad_quality(self):
        """
        @brief 计算
        """

        NC = self.number_of_cells()
        NN = self.number_of_nodes()
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        localEdge = self.ds.localEdge
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)

        l = np.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)
        c = mu*(1/(p*l) + 1/l2)

        val = np.zeros((NC, 3, 3), dtype=sefl.ftype)
        val[:, 0, 0] = c[:, 1] + c[:, 2]
        val[:, 0, 1] = -c[:, 2]
        val[:, 0, 2] = -c[:, 1]

        val[:, 1, 0] = -c[:, 2]
        val[:, 1, 1] = c[:, 0] + c[:, 2]
        val[:, 1, 2] = -c[:, 0]

        val[:, 2, 0] = -c[:, 1]
        val[:, 2, 1] = -c[:, 0]
        val[:, 2, 2] = c[:, 0] + c[:, 1]

        I = np.broadcast_to(cell[:, None, :], shape=(NC, 3, 3))
        J = np.broadcast_to(cell[:, :, None], shape=(NC, 3, 3))
        A = csr_matrix((val, (I, J)), shape=(NN, NN))

        cn = mu/area
        val[:, 0, 0] = 0
        val[:, 0, 1] = -cn
        val[:, 0, 2] = cn

        val[:, 1, 0] = cn
        val[:, 1, 1] = 0
        val[:, 1, 2] = -cn

        val[:, 2, 0] = -cn
        val[:, 2, 1] = cn
        val[:, 2, 2] = 0
        B = csr_matrix((val, (I, J)), shape=(NN, NN))
        return A, B

class TetRadiusRatio(QualityMetric):
    def quality(self, mesh):
        """
        @brief 计算半径比质量

        mu = R/(3*r)
        p = l_0 + l_1 + l_2
        q = l_0 * l_1 * l_2
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()

        localEdge = mesh.ds.localEdge
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
        
        l2 = np.zeros((NC, 3))
        s1 = np.sqrt(np.sum(np.cross(v[0],v[1])**2,axis=1))/2
        s2 = np.sqrt(np.sum(np.cross(v[0],v[2])**2,axis=1))/2
        s3 = np.sqrt(np.sum(np.cross(v[1],v[2])**2,axis=1))/2
        s4 = np.sqrt(np.sum(np.cross(v[3],v[4])**2,axis=1))/2
        s = s1+s2+s3+s4
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        d=l2[:,2,None]*np.cross(v[0],v[1])+l2[:,0,None]*np.cross(v[1],v[2])+l2[:,1,None]*np.cross(v[2],v[0])
        dm = np.sqrt(np.sum(d**2,axis=1))
        volume = np.abs(np.sum(np.cross(v[1], v[2])*v[0],axis=1))/6
        quality = (s*dm)/(108*volume**2)
        return quality

class QuadQuality(QualityMetric):
    def quality(self,mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        if np.max(node[:,2])-np.min(node[:,2])<1e-15:
            node = node[:,:2]
        NC = mesh.number_of_cells()
        localEdge = mesh.ds.localEdge
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
        if node.shape[1]==2:
            J = np.zeros((NC,4,2,2))
            J[:,0,0]=v[0]
            J[:,0,1]=-v[3]
            J[:,1,0]=v[1]
            J[:,1,1]=-v[0]
            J[:,2,0]=v[2]
            J[:,2,1]=-v[1]
            J[:,3,0]=v[3]
            J[:,3,1]=-v[2]
            detJ=np.linalg.det(J)
            flagJ0 = np.ones((NC,4),dtype=np.bool_)
            flagJ0[detJ<0]=False
            flagJ =np.ones(NC,dtype=np.bool_)
            flagJ[np.sum(flagJ0,axis=1)<4]=False
            quality = np.zeros((NC,4))
            for i in range(4):
                quality[:,i]=(np.sum(v[i]**2,axis=1)+np.sum(v[i-1]**2,axis=1))/(2*np.abs(np.cross(v[i],-v[i-1])))
            quality = np.sum(quality,axis=1)/4
            quality[flagJ==False]=0
        else:
            quality = np.zeros((NC,4))
            for i in range(4):
                quality[:,i]=(np.sum(v[i]**2,axis=1)+np.sum(v[i-1]**2,axis=1))/(2*np.sqrt(np.sum(np.cross(v[i],-v[i-1])**2,axis=1)))
            quality = np.sum(quality,axis=1)/4

        return quality
class HexQuality(QualityMetric):
    def quality(self,mesh):
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        
        NC = mesh.number_of_cells()
        localEdge = mesh.ds.localEdge
        v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
       
        quality = np.zeros((NC,8))
        l3 = np.zeros((NC,12))
        for i in range(12):
            l3[:, i] = np.sqrt(np.sum(v[i]**2, axis=1))*np.sum(v[i]**2,axis=1)
        quality[:,0]=(l3[:,0]+l3[:,3]+l3[:,4])/(3*np.sqrt(np.sum((np.cross(v[0],v[3])*v[4])**2,axis=1)))

        for i in range(1,4):
            quality[:,i]=(l3[:,i]+l3[:,i-1]+l3[:,i+4])/(3*np.sqrt(np.sum((np.cross(v[i],-v[i-1])*v[i+4])**2,axis=1)))
        
        quality[:,4]=(l3[:,8]+l3[:,11]+l3[:,4])/(3*np.sqrt(np.sum((np.cross(v[8],v[11])*(-v[4]))**2,axis=1)))
        for i in range(5,8):
            quality[:,i]=(l3[:,i+4]+l3[:,i+3]+l3[:,i])/(3*np.sqrt(np.sum((np.cross(v[i+4],-v[i+3])*(-v[i]))**2,axis=1)))
        quality=np.sum(quality,axis=1)/8
        return quality
