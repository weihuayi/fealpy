from typing import TypeVar, Generic

from ..backend import backend_manager as bm
from ..backend import TensorLike

from .opt import MeshCellQuality

class RadiusRatioQuality(MeshCellQuality):
    def __init__(self,mesh):
        self.mesh = mesh

    def fun(self,x:TensorLike) -> TensorLike:
        node = x
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()

        if self.mesh.top_dimension() == 2:
            localEdge = self.mesh.localEdge
            v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
            l2 = bm.zeros((NC, 3))
            for i in range(3):
                l2[:, i] = bm.sum(v[i]**2, axis=1)
            l = bm.sqrt(l2)
            p = l.sum(axis=1)
            q = l.prod(axis=1)
            area = bm.cross(v[2], -v[1])/2
            quality = p*q/(16*area**2)
        elif self.mesh.top_dimension() == 3:
            vji = node[cell[:,0]] - node[cell[:,1]]
            vki = node[cell[:,0]] - node[cell[:,2]]
            vmi = node[cell[:,0]] - node[cell[:,3]]

            vji2 = bm.sum(vji**2,axis=-1)
            vki2 = bm.sum(vki**2,axis=-1)
            vmi2 = bm.sum(vmi**2,axis=-1)

            d = vmi2[:,None]*(bm.cross(vji,vki))+vji2[:,None]*(bm.cross(vki,vmi))+vki2[:,None]*(bm.cross(vmi,vji))
            dl = bm.sqrt(bm.sum(d**2,axis=-1))
            face = self.mesh.face
            v01 = node[face[:,1],:] - node[face[:,0],:]
            v02 = node[face[:,2],:] - node[face[:,0],:]
            fm = bm.sqrt(bm.square(bm.cross(v01,v02)).sum(axis=1))/2.0

            cm = bm.sum(-vmi*bm.cross(vji,vki),axis=1)/6.0
            cell2face = self.mesh.cell_to_face()
            s_sum = bm.sum(fm[cell2face], axis=-1) 
            quality = s_sum*dl/108/cm/cm
        return quality

    def jac(self,x:TensorLike) -> TensorLike:
        node = x
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()

        if self.mesh.TD == 2:
            idxi = cell[:, 0]
            idxj = cell[:, 1] 
            idxk = cell[:, 2] 

            v0 = node[idxk] - node[idxj]
            v1 = node[idxi] - node[idxk]
            v2 = node[idxj] - node[idxi]

            area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
            l2 = bm.zeros((NC, 3), dtype=bm.float64)
            l2[:, 0] = bm.sum(v0**2, axis=1)
            l2[:, 1] = bm.sum(v1**2, axis=1)
            l2[:, 2] = bm.sum(v2**2, axis=1)
            l = bm.sqrt(l2)
            p = l.sum(axis=1, keepdims=True)
            q = l.prod(axis=1, keepdims=True)
            mu = p*q/(16*area**2)

            c = mu*(1/(p*l)+1/l2)
            cn = mu/area

            grad = bm.zeros((NC, 3, 2), dtype=bm.float64)
            cn = cn.reshape(-1)
            grad[:,0,0] = c[:, 1]*v1[:,0] - c[:, 2]*v2[:,0] + cn*v0[:,1]
            grad[:,0,1] = c[:, 1]*v1[:,1] - c[:, 2]*v2[:,1] - cn*v0[:,0]
            grad[:,1,0] = -c[:, 0]*v0[:,0] + c[:, 2]*v2[:,0] + cn*v1[:,1]
            grad[:,1,1] = -c[:, 0]*v0[:,1] + c[:, 2]*v2[:,1] - cn*v1[:,0]
            grad[:,2,0] = c[:, 0]*v0[:,0] - c[:, 1]*v1[:,0] + cn*v2[:,1]
            grad[:,2,1] = c[:, 0]*v0[:,1] - c[:, 1]*v1[:,1] - cn*v2[:,0]
            return grad

        elif self.mesh.TD == 3:
            v10 = node[cell[:, 0]] - node[cell[:, 1]]
            v20 = node[cell[:, 0]] - node[cell[:, 2]]
            v30 = node[cell[:, 0]] - node[cell[:, 3]]

            v21 = node[cell[:, 1]] - node[cell[:, 2]]
            v31 = node[cell[:, 1]] - node[cell[:, 3]]
            v32 = node[cell[:, 2]] - node[cell[:, 3]]

            l10 = bm.sum(v10**2, axis=-1)
            l20 = bm.sum(v20**2, axis=-1)
            l30 = bm.sum(v30**2, axis=-1)
            l21 = bm.sum(v21**2, axis=-1)
            l31 = bm.sum(v31**2, axis=-1)
            l32 = bm.sum(v32**2, axis=-1)

            d0 = bm.zeros((NC, 3), dtype=self.mesh.ftype)
            c12 =  bm.cross(v10, v20)
            d0 += l30[:, None]*c12
            c23 = bm.cross(v20, v30)
            d0 += l10[:, None]*c23
            c31 = bm.cross(v30, v10)
            d0 += l20[:, None]*c31

            c12 = bm.sum(c12*d0, axis=-1)
            c23 = bm.sum(c23*d0, axis=-1)
            c31 = bm.sum(c31*d0, axis=-1)
            c = c12 + c23 + c31

            A = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            A[:, 0, 0]  = 2*c
            A[:, 0, 1] -= 2*c23
            A[:, 0, 2] -= 2*c31
            A[:, 0, 3] -= 2*c12

            A[:, 1, 1] = 2*c23
            A[:, 2, 2] = 2*c31
            A[:, 3, 3] = 2*c12
            A[:, 1:, 0] = A[:, 0, 1:]
            
            K = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            K[:, 0, 1] -= l30 - l20
            K[:, 0, 2] -= l10 - l30
            K[:, 0, 3] -= l20 - l10
            K[:, 1:, 0] -= K[:, 0, 1:]

            K[:, 1, 2] -= l30
            K[:, 1, 3] += l20
            K[:, 2:, 1] -= K[:, 1, 2:]

            K[:, 2, 3] -= l10
            K[:, 3, 2] += l10

            S = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            face = self.mesh.entity('face')
            fv01 = node[face[:,1],:] - node[face[:,0],:]
            fv02 = node[face[:,2],:] - node[face[:,0],:]
            fm = bm.sqrt(bm.square(bm.cross(fv01,fv02)).sum(axis=1))/2.0
            #fm = self.mesh.entity_measure("face")
            #cm = self.mesh.entity_measure("cell")

            cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
            c2f = self.mesh.cell_to_face()

            s = fm[c2f]
            s_sum = bm.sum(s, axis=-1)
             
            p0 = (l31/s[:,2] + l21/s[:,3] + l32/s[:,1])/4
            p1 = (l32/s[:,0] + l20/s[:,3] + l30/s[:,2])/4
            p2 = (l30/s[:,1] + l10/s[:,3] + l31/s[:,0])/4
            p3 = (l10/s[:,2] + l20/s[:,1] + l21/s[:,0])/4

            q10 = -(bm.sum(v31*v30, axis=-1)/s[:,2]+bm.sum(v21*v20, axis=-1)/s[:,3])/4
            q20 = -(bm.sum(v32*v30, axis=-1)/s[:,1]+bm.sum(-v21*v10, axis=-1)/s[:,3])/4
            q30 = -(bm.sum(-v32*v20, axis=-1)/s[:,1]+bm.sum(-v31*v10, axis=-1)/s[:,2])/4
            q21 = -(bm.sum(v32*v31, axis=-1)/s[:,0]+bm.sum(v20*v10, axis=-1)/s[:,3])/4
            q31 = -(bm.sum(v30*v10, axis=-1)/s[:,2]+bm.sum(-v32*v21, axis=-1)/s[:,0])/4
            q32 = -(bm.sum(v31*v21, axis=-1)/s[:,0]+bm.sum(v30*v20, axis=-1)/s[:,1])/4
            
            S[:, 0, 0] = p0
            S[:, 0, 1] = q10
            S[:, 0, 2] = q20
            S[:, 0, 3] = q30
            S[:, 1:,0] = S[:, 0, 1:]

            S[:, 1, 1] = p1
            S[:, 1, 2] = q21
            S[:, 1, 3] = q31
            S[:, 2:,1] = S[:, 1, 2:]

            S[:, 2, 2] = p2
            S[:, 2, 3] = q32
            S[:, 3, 2] = q32
            S[:, 3, 3] = p3
            
            C0 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            C1 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            C2 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            
            def f(CC, xx):
                CC[:, 0, 1] = xx[:, 2]
                CC[:, 0, 2] = xx[:, 3]
                CC[:, 0, 3] = xx[:, 1]
                CC[:, 1, 0] = xx[:, 3]
                CC[:, 1, 2] = xx[:, 0]
                CC[:, 1, 3] = xx[:, 2]
                CC[:, 2, 0] = xx[:, 1]
                CC[:, 2, 1] = xx[:, 3]
                CC[:, 2, 3] = xx[:, 0]
                CC[:, 3, 0] = xx[:, 2]
                CC[:, 3, 1] = xx[:, 0]
                CC[:, 3, 2] = xx[:, 1]

            f(C0, node[cell, 0])
            f(C1, node[cell, 1])
            f(C2, node[cell, 2])

            C0 = 0.5*(-C0 + C0.swapaxes(-1, -2))
            C1 = 0.5*(C1  - C1.swapaxes(-1, -2))
            C2 = 0.5*(-C2 + C2.swapaxes(-1, -2))
            
            B0 = -d0[:,0,None,None]*K
            B1 = d0[:,1,None,None]*K
            B2 = -d0[:,2,None,None]*K
            
            ld0 = bm.sum(d0**2,axis=-1)
     
            A  /= ld0[:,None,None]
            B0 /= ld0[:,None,None]
            B1 /= ld0[:,None,None]
            B2 /= ld0[:,None,None]

            S  /= s_sum[:,None,None]

            C0 /= 3*cm[:,None,None]
            C1 /= 3*cm[:,None,None]
            C2 /= 3*cm[:,None,None]

            A  += S
            B0 -= C0
            B1 -= C1
            B2 -= C2

            mu = s_sum*bm.sqrt(ld0)/(108*cm**2)
            #mu = s_sum*np.sqrt(ld0)/(108)

            A  *= mu[:,None,None]/NC
            B0 *= mu[:,None,None]/NC
            B1 *= mu[:,None,None]/NC
            B2 *= mu[:,None,None]/NC

            cell_node = node[cell]
            grad = bm.zeros((NC, 4, 3), dtype=bm.float64)

            grad[:, :, 0]=bm.einsum('ijk,ik->ij',A,cell_node[:,:,0])+bm.einsum('ijk,ik->ij',B2,cell_node[:,:,1])+bm.einsum('ijk,ik->ij',B1,cell_node[:,:,2])

            grad[:, :, 1]=bm.einsum('ijk,ik->ij', A,cell_node[:,:,1])-bm.einsum('ijk,ik->ij',B2,cell_node[:,:,0])+bm.einsum('ijk,ik->ij',B0,cell_node[:,:,2])

            grad[:, :, 2]=bm.einsum('ijk,ik->ij', A,cell_node[:,:,2])-bm.einsum('ijk,ik->ij',B1,cell_node[:,:,0])-bm.einsum('ijk,ik->ij',B0,cell_node[:,:,1])
            return grad

    def hess(self,x:TensorLike):
        #node = self.mesh.entity('node')
        node = x
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()

        if self.mesh.TD == 2:
            idxi = cell[:, 0]
            idxj = cell[:, 1] 
            idxk = cell[:, 2] 

            v0 = node[idxk] - node[idxj]
            v1 = node[idxi] - node[idxk]
            v2 = node[idxj] - node[idxi]

            area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
            l2 = bm.zeros((NC, 3), dtype=bm.float64)
            l2[:, 0] = bm.sum(v0**2, axis=1)
            l2[:, 1] = bm.sum(v1**2, axis=1)
            l2[:, 2] = bm.sum(v2**2, axis=1)
            l = bm.sqrt(l2)
            p = l.sum(axis=1, keepdims=True)
            q = l.prod(axis=1, keepdims=True)
            mu = p*q/(16*area**2)

            c = mu*(1/(p*l)+1/l2)
            cn = mu/area

            A = bm.zeros((NC, 3, 3), dtype=bm.float64)
            B = bm.zeros((NC, 3, 3), dtype=bm.float64)
            A[:, 0, 0] = c[:, 1] + c[:, 2]
            A[:, 0, 1] = -c[:, 2]
            A[:, 0, 2] = -c[:, 1]
            A[:, 1, 0] = -c[:, 2]
            A[:, 1, 1] = c[:, 0] + c[:, 2]
            A[:, 1, 2] = -c[:, 0]
            A[:, 2, 0] = -c[:, 1]
            A[:, 2, 1] = -c[:, 0]
            A[:, 2, 2] = c[:, 0] + c[:, 1]

            B[:, 0, 1] = B[:, 1, 2] = B[:, 2, 0] = -cn.reshape(-1)
            B[:, 1, 0] = B[:, 2, 1] = B[:, 0, 2] = cn.reshape(-1)
            return (A,B)

        elif self.mesh.TD == 3:
            v10 = node[cell[:, 0]] - node[cell[:, 1]]
            v20 = node[cell[:, 0]] - node[cell[:, 2]]
            v30 = node[cell[:, 0]] - node[cell[:, 3]]

            v21 = node[cell[:, 1]] - node[cell[:, 2]]
            v31 = node[cell[:, 1]] - node[cell[:, 3]]
            v32 = node[cell[:, 2]] - node[cell[:, 3]]

            l10 = bm.sum(v10**2, axis=-1)
            l20 = bm.sum(v20**2, axis=-1)
            l30 = bm.sum(v30**2, axis=-1)
            l21 = bm.sum(v21**2, axis=-1)
            l31 = bm.sum(v31**2, axis=-1)
            l32 = bm.sum(v32**2, axis=-1)

            d0 = bm.zeros((NC, 3), dtype=self.mesh.ftype)
            c12 =  bm.cross(v10, v20)
            d0 += l30[:, None]*c12
            c23 = bm.cross(v20, v30)
            d0 += l10[:, None]*c23
            c31 = bm.cross(v30, v10)
            d0 += l20[:, None]*c31

            c12 = bm.sum(c12*d0, axis=-1)
            c23 = bm.sum(c23*d0, axis=-1)
            c31 = bm.sum(c31*d0, axis=-1)
            c = c12 + c23 + c31

            A = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            A[:, 0, 0]  = 2*c
            A[:, 0, 1] -= 2*c23
            A[:, 0, 2] -= 2*c31
            A[:, 0, 3] -= 2*c12

            A[:, 1, 1] = 2*c23
            A[:, 2, 2] = 2*c31
            A[:, 3, 3] = 2*c12
            A[:, 1:, 0] = A[:, 0, 1:]
            
            K = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            K[:, 0, 1] -= l30 - l20
            K[:, 0, 2] -= l10 - l30
            K[:, 0, 3] -= l20 - l10
            K[:, 1:, 0] -= K[:, 0, 1:]

            K[:, 1, 2] -= l30
            K[:, 1, 3] += l20
            K[:, 2:, 1] -= K[:, 1, 2:]

            K[:, 2, 3] -= l10
            K[:, 3, 2] += l10

            S = bm.zeros((NC, 4, 4), dtype=self.mesh.ftype)
            face = self.mesh.entity('face')
            fv01 = node[face[:,1],:] - node[face[:,0],:]
            fv02 = node[face[:,2],:] - node[face[:,0],:]
            fm = bm.sqrt(bm.square(bm.cross(fv01,fv02)).sum(axis=1))/2.0
            #fm = self.mesh.entity_measure("face")
            #cm = self.mesh.entity_measure("cell")

            cm = bm.sum(-v30*bm.cross(v10,v20),axis=1)/6.0
            c2f = self.mesh.cell_to_face()

            s = fm[c2f]
            s_sum = bm.sum(s, axis=-1)
             
            p0 = (l31/s[:,2] + l21/s[:,3] + l32/s[:,1])/4
            p1 = (l32/s[:,0] + l20/s[:,3] + l30/s[:,2])/4
            p2 = (l30/s[:,1] + l10/s[:,3] + l31/s[:,0])/4
            p3 = (l10/s[:,2] + l20/s[:,1] + l21/s[:,0])/4

            q10 = -(bm.sum(v31*v30, axis=-1)/s[:,2]+bm.sum(v21*v20, axis=-1)/s[:,3])/4
            q20 = -(bm.sum(v32*v30, axis=-1)/s[:,1]+bm.sum(-v21*v10, axis=-1)/s[:,3])/4
            q30 = -(bm.sum(-v32*v20, axis=-1)/s[:,1]+bm.sum(-v31*v10, axis=-1)/s[:,2])/4
            q21 = -(bm.sum(v32*v31, axis=-1)/s[:,0]+bm.sum(v20*v10, axis=-1)/s[:,3])/4
            q31 = -(bm.sum(v30*v10, axis=-1)/s[:,2]+bm.sum(-v32*v21, axis=-1)/s[:,0])/4
            q32 = -(bm.sum(v31*v21, axis=-1)/s[:,0]+bm.sum(v30*v20, axis=-1)/s[:,1])/4
            
            S[:, 0, 0] = p0
            S[:, 0, 1] = q10
            S[:, 0, 2] = q20
            S[:, 0, 3] = q30
            S[:, 1:,0] = S[:, 0, 1:]

            S[:, 1, 1] = p1
            S[:, 1, 2] = q21
            S[:, 1, 3] = q31
            S[:, 2:,1] = S[:, 1, 2:]

            S[:, 2, 2] = p2
            S[:, 2, 3] = q32
            S[:, 3, 2] = q32
            S[:, 3, 3] = p3
            
            C0 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            C1 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            C2 = bm.zeros((NC, 4, 4), dtype=bm.float64)
            
            def f(CC, xx):
                CC[:, 0, 1] = xx[:, 2]
                CC[:, 0, 2] = xx[:, 3]
                CC[:, 0, 3] = xx[:, 1]
                CC[:, 1, 0] = xx[:, 3]
                CC[:, 1, 2] = xx[:, 0]
                CC[:, 1, 3] = xx[:, 2]
                CC[:, 2, 0] = xx[:, 1]
                CC[:, 2, 1] = xx[:, 3]
                CC[:, 2, 3] = xx[:, 0]
                CC[:, 3, 0] = xx[:, 2]
                CC[:, 3, 1] = xx[:, 0]
                CC[:, 3, 2] = xx[:, 1]

            f(C0, node[cell, 0])
            f(C1, node[cell, 1])
            f(C2, node[cell, 2])

            C0 = 0.5*(-C0 + C0.swapaxes(-1, -2))
            C1 = 0.5*(C1  - C1.swapaxes(-1, -2))
            C2 = 0.5*(-C2 + C2.swapaxes(-1, -2))
            
            B0 = -d0[:,0,None,None]*K
            B1 = d0[:,1,None,None]*K
            B2 = -d0[:,2,None,None]*K
            
            ld0 = bm.sum(d0**2,axis=-1)
     
            A  /= ld0[:,None,None]
            B0 /= ld0[:,None,None]
            B1 /= ld0[:,None,None]
            B2 /= ld0[:,None,None]

            S  /= s_sum[:,None,None]

            C0 /= 3*cm[:,None,None]
            C1 /= 3*cm[:,None,None]
            C2 /= 3*cm[:,None,None]

            A  += S
            B0 -= C0
            B1 -= C1
            B2 -= C2

            mu = s_sum*bm.sqrt(ld0)/(108*cm**2)
            #mu = s_sum*np.sqrt(ld0)/(108)

            #''' 
            A  *= mu[:,None,None]/NC
            B0 *= mu[:,None,None]/NC
            B1 *= mu[:,None,None]/NC
            B2 *= mu[:,None,None]/NC
            return (A,B0,B1,B2)






        

