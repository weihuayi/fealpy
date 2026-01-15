from ..backend import bm
from ..backend import TensorLike
from ..decorator import barycentric, cartesian

from ..mesh import Mesh
from .space import FunctionSpace

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

import matplotlib.pyplot as plt

_MT = TypeVar('_MT', bound=Mesh)

class ScaledMonomialSpace3d(FunctionSpace, Generic[_MT]):
    """
    3D scaled monomial function space.
    """
    def __init__(self, mesh: _MT, p : int  , q : int = None , bc = None):
        """
        Parameters
            mesh : TetrahedronMesh , PolyhedronMesh or HalfEdgeMesh3d object
            p : int
                The space degree 
            q : int, optional
                the index of the quadrature formula to use.
            bc : optional
                cell barycenter, shape:(NC, 3)
        """
        self.p = p
        self.GD = 3
        self.mesh = mesh
        mtype = mesh.meshtype
        
        self.ikwargs = bm.context(mesh.cell[0]) if mtype =='polyhedron' else bm.context(mesh.cell)
        self.fkwargs = bm.context(mesh.node)
        
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.facebarycenter = mesh.entity_barycenter('face')
        
        self.q = q if q is not None else p + 3
        self.cm = self.mesh.entity_measure('cell')
        self.fm = self.mesh.entity_measure('face')
        self.em = self.mesh.entity_measure('edge')

        self.csize = self.cm**(1/3)
        self.fsize = self.fm**(1/2)
        self.esize = self.em

        n = self.mesh.face_unit_normal()
        a, _, self.faceframe = bm.linalg.svd(n[:, None ,:])
        a =  a.reshape(-1)
        id0 = a == 1
        id1 = a == -1        
        self.faceframe = bm.set_at(self.faceframe , (id0,2,...) , -self.faceframe[id0,2,...])
        self.faceframe = bm.set_at(self.faceframe , (id1,...) , -self.faceframe[id1,...])

    def geo_dimension(self):
        return self.GD
    
    def cell_to_dof(self, p = None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cdof = self.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = bm.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof
    
    def face_to_dof(self, p = None):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        fdof = self.number_of_local_dofs(p=p, doftype='face')
        face2dof = bm.arange(NF*fdof).reshape(NF, fdof)
        return face2dof
    
    def number_of_local_dofs(self, p=None, doftype='cell'):
        p = self.p if p is None else p
        if doftype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif doftype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif doftype in {'edge', 1}:
            return p+1
        elif doftype in {'node', 0}:
            return 0
        
    def number_of_global_dofs(self, p=None, doftype='cell'):
        ldof = self.number_of_local_dofs(p=p, doftype=doftype)
        if doftype in {'cell', 3}:
            N = self.mesh.number_of_cells()
        elif doftype in {'face', 2}:
            N = self.mesh.number_of_faces()
        return N*ldof
    
    def diff_index_1(self, p=None):
        """
        Compute the first derivative index.
        The indices and coefficients of the non-zero terms after taking the first derivative of the basis functions.
        Parameters:
              p : int the degree
        Returns:
              dict : {'x':(index, coefficient), 
                      'y':(index, coefficient),
                      'z':(index, coefficient)}
        """
        p = self.p if p is None else p
        index = bm.multi_index_matrix(p,3)
        
        x, = bm.nonzero(index[:, 1] > 0)
        y, = bm.nonzero(index[:, 2] > 0)
        z, = bm.nonzero(index[:, 3] > 0)

        return {'x':(x, index[x, 1]), 
                'y':(y, index[y, 2]),
                'z':(z, index[z, 3])
                }
        
    def diff_index_2(self, p=None):
        """
        Compute the second derivative index.
        The indices and coefficients of the non-zero terms after taking the second derivative of the basis functions.
        Parameters:
            p : int the degree
        Returns:
            dict : {'xx':(index, coefficient), 
                    'yy':(index, coefficient), 
                    'zz':(index, coefficient),
                    'xy':(index, coefficient),
                    'xz':(index, coefficient),
                    'yz':(index, coefficient)}
        """
        p = self.p if p is None else p
        index = bm.multi_index_matrix(p,3)

        xx, = bm.nonzero(index[:, 1] > 1)
        yy, = bm.nonzero(index[:, 2] > 1)
        zz, = bm.nonzero(index[:, 3] > 1)

        xy, = bm.nonzero((index[:, 1] > 0) & (index[:, 2] > 0))
        xz, = bm.nonzero((index[:, 1] > 0) & (index[:, 3] > 0))
        yz, = bm.nonzero((index[:, 2] > 0) & (index[:, 3] > 0))

        return {'xx':(xx, index[xx, 1]*(index[xx, 1]-1)), 
                'yy':(yy, index[yy, 2]*(index[yy, 2]-1)),
                'zz':(zz, index[zz, 3]*(index[zz, 3]-1)),
                'xy':(xy, index[xy, 1]*index[xy, 2]),
                'xz':(xz, index[xz, 1]*index[xz, 3]),
                'yz':(yz, index[yz, 2]*index[yz, 3])
                }

    def face_index_1(self, p=None):
        """
        Compute the first derivative index.
        The indices and coefficients of the non-zero terms after taking the first derivative of the basis functions.
        Parameters:
              p : int the degree
        Returns:
              dict : {'x':(index, coefficient), 
                      'y':(index, coefficient),
                      'z':(index, coefficient)}
        """
        p = self.p if p is None else p
        index = bm.multi_index_matrix(p,2)

        x, = bm.nonzero(index[:, 0] > 0)
        y, = bm.nonzero(index[:, 1] > 0)
        z, = bm.nonzero(index[:, 2] > 0)
        
        return {'x': x, 'y':y, 'z':z}

    def partial_matrix(self, p=None, index=_S):
        """
        Compute the partial derivative matrix. It is a linear mapping.
        partial m = mP 
        
        Parameters:
            p : int, optional
                The degree of the polynomial space. If None, use the default degree of the space.
            index : slice or ndarray, optional
                The index of the cells to compute the partial derivative matrix.   
        Returns:
            Px, Py ,Pz: tuple of ndarray  
        """
        p = p or self.p
        mindex = bm.multi_index_matrix(p, 3) 
        N = len(mindex)
        NC = self.mesh.number_of_cells()
        h = self.csize

        I, = bm.where(mindex[:, 1] > 0)
        Px = bm.zeros([NC, N, N], **self.fkwargs)
        Px = bm.set_at(Px, (...,bm.arange(len(I)), I), mindex[None, I, 1]/h[:, None])

        I, = bm.where(mindex[:, 2] > 0)
        Py = bm.zeros([NC, N, N], **self.fkwargs)
        Py = bm.set_at(Py, (...,bm.arange(len(I)), I), mindex[None, I, 2]/h[:, None])

        I, = bm.where(mindex[:, 3] > 0)
        Pz = bm.zeros([NC, N, N], **self.fkwargs)
        Pz = bm.set_at(Pz, (...,bm.arange(len(I)), I), mindex[None, I, 3]/h[:, None])
        return Px[index], Py[index], Pz[index]
    
    @cartesian
    def basis(self, point , index = _S ,p = None ):
        """
        Compute the basis values at point
        
        Parameters:
            point : ndarray
                The shape of point is (NC, ..., 2)
                the points should be on the cell and always quadrature points
            index : slice or ndarray
            p : int
        Returns:
            phi : ndarray
                The shape of `phi` is (NC, ..., ldof)
        """
        p = self.p if p is None else p
        h = self.csize
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0], **self.fkwargs).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = bm.ones(shape, **self.fkwargs)  # (NC,...,ldof)
        phi = bm.set_at(phi,(..., slice(1,4)),(point - self.cellbarycenter[index])/h[index].reshape(-1, 1))
        if p > 1:
            start = 4
            for i in range(2, p+1):
                n = (i+1)*i//2
                phi = bm.set_at(phi , (..., slice(start, start+n)), phi[..., start-n:start]*phi[..., [1]])
                phi = bm.set_at(phi , (..., slice(start+n, start+n+i)) , phi[..., start-i:start]*phi[..., [2]])
                phi = bm.set_at(phi , (..., start + n + i), phi[..., start-1]*phi[..., 3])
                start += n + i + 1  
        return phi
    
    @cartesian
    def face_basis(self,point , index = _S , p = None):
        """
        Compute the basis values at point on each face 

        Parameters
        point : ndarray
            The shape of `point` is (NF,...,3), NC is the number of cells

        Returns
        phi : ndarray
            The shape of `phi` is (NF,...,fdof)

        Notes
        -----
        The `faceframe` is local orthogonal coordinate frame system.  
        `faceframe[i, 0, :]` is the fixed unit norm vector of i-th face. 
        `faceframe[i, 1:3, :]` are the two unit tangent vector on i-th face. 

        Dot the 3d vector `point - facebarycenter` with `faceframe[i, 1:, :]`,
        repectively, one can get the local coordinate component on i-th face.
        """
        p = self.p if p is None else p
        h = self.fsize
        bc = self.facebarycenter
        frame = self.faceframe
        
        fdof = self.number_of_local_dofs(p=p, doftype='face')
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0],**self.fkwargs).reshape(shape)

        shape = point.shape[:-1]+(fdof,)
        phi = bm.ones(shape, **self.fkwargs)  # (NF, ... ,fdof)
        p2 = (point - self.facebarycenter[index])/h[index].reshape(-1, 1)
        phi = bm.set_at(phi, (..., slice(1, 3)),bm.einsum('...jk, jnk->...jn', p2, frame[index, 1:, :]))
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi = bm.set_at(phi, (..., slice(start, start+i)), phi[..., start-i:start]*phi[..., [1]])
                phi = bm.set_at(phi, (..., start+i), phi[..., start-1]*phi[..., 2])
                start += i+1
        return phi
    
    @cartesian
    def edge_basis(self, point, index=_S, p=None):
        """
        Compute the basis values at point on each edge.

        Parameters
            point : ndarray
                The shape of `point` is (NE,...,3), NE is the number of edges
            index : int
                The index of the edge
            p : int
                The polynomial degree

        Returns
            phi : ndarray
                The shape of `phi` is (NE,...,edof)

        """
        p = self.p if p is None else p
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0], **self.fkwargs).reshape(shape)

        ebc = self.mesh.entity_barycenter('edge')
        eh = self.esize
        et = self.mesh.edge_tangent(unit= True)
        val = bm.sum((point - ebc[index])*et[index], axis=-1)/eh[index]
        phi = bm.ones(val.shape + (p+1,), **self.fkwargs)
        if p == 1:
            phi = bm.set_at(phi, (..., 1), val)
        else:
            phi = bm.set_at(phi, (..., 1), val[..., bm.newaxis])
            bm.cumprod(phi, axis=-1, out=phi)
        return phi
    
    @cartesian
    def grad_basis(self, point, index=_S, p=None , scaled=True):
        """
        Compute the gradient of the basis functions at a set of 'point'
        
        Parameters:
            point : ndarray
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
            p : int
            scaled : bool if True, return the scaled gradient
        Returns:
            gphi : ndarray
                The shape of gphi is (NC, ..., ldof, 3)
        """
        p = self.p if p is None else p
        h = self.csize
        num = len(h) if type(index) is slice else len(index)
 
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 3)
        phi = self.basis(point, index=index, p=p-1)
        idx = self.diff_index_1(p=p)
        gphi = bm.zeros(shape, **self.fkwargs)
        x = idx['x']
        y = idx['y']
        z = idx['z']
        gphi = bm.set_at(gphi, (..., x[0], 0), bm.einsum('i, ...i->...i', x[1], phi))
        gphi = bm.set_at(gphi, (..., y[0], 1), bm.einsum('i, ...i->...i', y[1], phi))
        gphi = bm.set_at(gphi, (..., z[0], 2), bm.einsum('i, ...i->...i', z[1], phi))

        if scaled:
            if point.shape[-2] == num:
                return gphi/h[index].reshape(-1, 1, 1)
            elif point.shape[0] == num:
                return gphi/h[index].reshape(-1, 1, 1, 1)
        else:
            return gphi
        
    @cartesian
    def laplace_basis(self, point, index=_S, p=None , scaled=True):
        """
        Compute the value of the laplace of the basis at a set of 'point'
        
        Parameters:
            point : numpy array
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
            p : int
            scaled : bool
                if True, return the scaled laplace
        Returns:
            lphi : numpy array
                the shape of lphi is (NC,..., ldof)
        """
        p = self.p if p is None else p
        area = self.cm
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof,)
        lphi = bm.zeros(shape, **self.fkwargs)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            xx = idx['xx']
            yy = idx['yy']
            zz = idx['zz']
            lphi = bm.index_add(lphi, (..., xx[0]), bm.einsum('i, ...i->...i', xx[1], phi))
            lphi = bm.index_add(lphi, (..., yy[0]), bm.einsum('i, ...i->...i', yy[1], phi))
            lphi = bm.index_add(lphi, (..., zz[0]), bm.einsum('i, ...i->...i', zz[1], phi))

        if scaled:
            return lphi/area[index].reshape((-1,)+(1,)*(point.ndim-1))
        else:
            return lphi

    @cartesian
    def hessian_basis(self, point, index=_S, p=None ,scaled =True):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters:
            point : numpy array
                The shape of point is (NC, ..., 2)
            index : slice or ndarray
            p : int
        Returns:
            hphi : numpy array
                the shape of hphi is (NC,..., ldof, 2, 2)
        """
        p = self.p if p is None else p

        area = self.cm
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 3, 3)
        hphi = bm.zeros(shape, **self.fkwargs)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            xx = idx['xx']
            yy = idx['yy']
            zz = idx['zz']
            xy = idx['xy']
            xz = idx['xz']
            yz = idx['yz']
            hphi = bm.set_at(hphi, (..., xx[0], 0, 0), bm.einsum('i, ...i->...i', xx[1], phi))
            hphi = bm.set_at(hphi, (..., xy[0], 0, 1), bm.einsum('i, ...i->...i', xy[1], phi))
            hphi = bm.set_at(hphi, (..., xz[0], 0, 2), bm.einsum('i, ...i->...i', xz[1], phi))
            hphi = bm.set_at(hphi, (..., yy[0], 1, 1), bm.einsum('i, ...i->...i', yy[1], phi))
            hphi = bm.set_at(hphi, (..., yz[0], 1, 2), bm.einsum('i, ...i->...i', yz[1], phi))
            hphi = bm.set_at(hphi, (..., zz[0], 2, 2), bm.einsum('i, ...i->...i', zz[1], phi))
            hphi = bm.set_at(hphi, (..., 1, 0), hphi[..., 0, 1])
        if scaled:
            return hphi/area[index].reshape((-1,)+(1,)*int(hphi.ndim-1))
        else:
            return hphi

    def grad_m_basis(self, m, point, index=_S, p=None, scaled=True):
        """!
        @brief m=3时导数排列顺序: [xxx, xxy, xxz, xyx, xyy , xyz , xzx , xzy, xzz,
                                   yxx, yxy , yxz ,yyx, yyy, yyz , yzx,  yzy, yzz,
                                   zxx, zxy , zxz , zyx,  zyy, zyz , zzx , zzy, zzz]
        """
        phi = self.basis(point, index=index, p=p)
        gmphi = bm.zeros(phi.shape+(3**m, ), **self.fkwargs)
        P = self.partial_matrix(index=index)

        #f = lambda x: bm.array([int(ss) for ss in bm.binary_repr(x, m)], dtype=bm.int32)
        #idx = bm.array(list(map(f, bm.arange(2**m))))
        def to_binary_array(x, m):
            # 获取 x 的二进制位，确保长度为 m
            return bm.tensor([((x >> (m - i - 1)) & 1) for i in bm.arange(m,dtype=bm.int32)],**self.ikwargs)
        idx = bm.stack([to_binary_array(x, m) for x in bm.arange(3**m,**self.ikwargs)])
        for i in range(3**m):
            M = bm.copy(P[idx[i, 0]])
            for j in range(1, m):
                M = bm.einsum("cij, cjk->cik", M, P[idx[i, j]])
            gmphi = bm.set_at(gmphi, (..., i),bm.einsum('cli, c...l->c...i', M, phi))
        return gmphi
    
    @cartesian
    def value(self, uh, point, index=_S):
        """
        Compute the value of the finite element function at a set of 'point'
        
        Parameters:
            uh : Function
            point : ndarray
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
                The index of the cells to compute the value.
        Returns:
            value : ndarray
                The shape of value is (NC, ...)
        """
        phi = self.basis(point, index=index)
        cell2dof = self.cell_to_dof()[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        return bm.einsum(s1, phi, uh[cell2dof], **self.fkwargs)
    
    @cartesian
    def grad_value(self, uh, point, index=_S):
        """
        Compute the gradient of the finite element function at a set of 'point'

        Parameters:
            uh : Function
            point : ndarray
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
                The index of the cells to compute the gradient.
        Returns:
            grad : ndarray
                The shape of grad is (NC, ..., 3)
        """
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.cell_to_dof()[index]
        if (type(index) is TensorLike) and (index.dtype.name == 'bool'):
            N = bm.sum(index)
        elif type(index) is slice:
            N = len(cell2dof)
        else:
            N = len(index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == N:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return bm.einsum(s1, gphi, uh[cell2dof[index]])
        elif point.shape[0] == N:
            return bm.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]])
        
    @cartesian
    def laplace_value(self, uh, point, index=_S):
        """
        Compute the laplace of the finite element function at a set of 'point'
        
        Parameters:
            uh : Function
            point : ndarray
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
                The index of the cells to compute the laplace.
        Returns:
            laplace_value : ndarray
                The shape of laplace_value is (NC, ...)
        """
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.cell_to_dof()[index]
        return bm.einsum('...ij, ij->...i', lphi, uh[cell2dof])
    
    @cartesian
    def hessian_value(self, uh, point, index=_S):
        """
        Compute the hessian of the finite element function at a set of 'point'
        Parameters:
            uh : Function
            point : ndarray
                The shape of point is (NC, ..., 3)
            index : slice or ndarray
                The index of the cells to compute the hessian.
        Returns:
            hessian_value : ndarray
                The shape of hessian_value is (NC, ..., 3, 3)
        """
        hphi = self.hessian_basis(point, index=index) #(NC,NQ,ldof, 3, 3)
        cell2dof = self.cell_to_dof()
        return bm.einsum('c...lij, cl->c...ij', hphi, uh[cell2dof[index]])
    
    def face_integral(self, f):
        """
        Compute the integral of a function on the edges of the mesh.
        
        Parameters:
            f : callable
                The function to integrate. It should have the signature f(x, index),
        Returns:
            ndarray : The integral of the function on each cell. shape is (NC, ...)
        """
        mesh = self.mesh
        p = self.p
        node = mesh.node
        face = mesh.face
        face2cell = mesh.face_to_cell()

        isInFace = (face2cell[:, 0] != face2cell[:, 1])

        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(p+3, etype='face', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, NFV)  (NQ,)
        ps = bm.einsum('ij, kjm->kim', bcs, node[face]) # (NQ, NFV) (NF, NFV, 3) -> (NF, NQ, 3)
        f1 = f(ps, index=face2cell[:, 0]) # (NF, NQ, ...)
        fm = self.fm
        H0 = bm.einsum('eq..., q, e-> e...', f1, ws, fm) # (NF,...)
        f2 = f(ps, index=face2cell[:, 1])
        H1 = bm.einsum('eq..., q, e-> e...', f2[isInFace], ws, fm[isInFace]) # (inNF,...)
        H = bm.zeros((NC,)+ f1.shape[2:], **self.fkwargs)
        bm.index_add(H, face2cell[:, 0], H0)
        bm.index_add(H, face2cell[isInFace, 1], H1)
        return H
    
    def integral(self, f):
        """
        The integration process of homogeneous functions is applicable to arbitrary polygonal meshes,
        where the volume integral is transformed into a boundary integral.
        
        Parameters:
            f : callable
                The function to integrate. It should have the signature f(x, index),      
        Returns:
            ndarray : The integral of the function on each cell. shape is (NC, ...)
        """
        mesh = self.mesh
        p = self.p
        node = mesh.node
        face = mesh.face
        face2cell = mesh.face_to_cell()
        facebarycenter = self.facebarycenter
        cellbarycenter = self.cellbarycenter

        isInFace = (face2cell[:, 0] != face2cell[:, 1])

        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(p+3, etype='face', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, NFV)  (NQ,)
        ps = bm.einsum('ij, kjm->kim', bcs, node[face]) # (NQ, NFV) (NF, NFV, 3) -> (NF, NQ, 3)
        f1 = f(ps, index=face2cell[:, 0]) # (NF, NQ, fldof)
        nm = mesh.face_normal()
        b = node[face[:, 0]] - cellbarycenter[face2cell[:, 0]]
        H0 = bm.einsum('eq..., q, ed, ed-> e...', f1, ws, b, nm) # (NF,...)
        f2 = f(ps, index=face2cell[:, 1])
        b = node[face[:, 1]] - cellbarycenter[face2cell[:, 1]]
        H1 = bm.einsum('eq..., q, ed, ed-> e...', f2, ws, b, -nm) # (inNF,...)
        H = bm.zeros((NC,)+ f1.shape[2:], **self.fkwargs)
        bm.index_add(H, face2cell[:, 0], H0)
        bm.index_add(H, face2cell[isInFace, 1], H1)
        multiIndex = bm.multi_index_matrix(p=p)
        q = bm.sum(multiIndex, axis=1)
        if H.ndim == 2:
            H /= q+3
        else:
            H /= q + q.reshape(-1, 1) + 3
        return H
    
    def cell_mass_matrix(self,p = None):
        """
        Cell mass matrix, shape:(NC, ldof, ldof)
        
        Parameters:
            p : int, optional
                The degree of the polynomial space. If None, use the default degree of the space.
        Returns:
            ndarray : The cell mass matrix. shape is (NC, ldof, ldof)
        """
        p = self.p if p is None else p
        def f(x, index):
            phi = self.basis(x, index=index, p=p)
            return bm.einsum('eqi, eqj -> eqij', phi, phi)
        return self.integral(f) # 积分
    
    def face_mass_matrix(self, p=None):
        """
        Face mass matrix, shape:(NF, fldof, fldof)
        
        Parameters:
            p : int, optional
                The degree of the polynomial space. If None, use the default degree of the space.
        Returns:
            ndarray : The cell mass matrix. shape is (NF, fldof, fldof)
        """
        p = self.p if p is None else p
        mesh = self.mesh
        fm = self.fm
        qf = mesh.quadrature_formula(p+3, etype="face", qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # NQ
        ps = self.mesh.face_bc_to_point(bcs) # (NF, NQ, 3)
        phi = self.face_basis(ps, p=p) #(NF,NQ,fldof)
        H = bm.einsum('q, fqk, fqm, f->fkm', ws, phi, phi, fm)
        return H
    
    def face_cell_mass_matrix(self, p=None, cp=None):
        """
        Compute the mixed mass matrix for cells and cell faces.
        Parameters:
            p : int, optional
                The degree of the polynomial space for faces. If None, use the default degree of the space.
            cp : int, optional
                The degree of the polynomial space for cells. If None, use p+1.
        Returns:
            LM, RM : tuple of ndarray
                The left and right mixed mass matrices. Each has shape (NC, NQ, fldof, cldof),
                where NE is the number of edges, NQ is the number of quadrature points,
                fldof is the number of local degrees of freedom for faces, and cldof is the number of local degrees of freedom for cells.
        """
        p = self.p if p is None else p
        cp = p+1 if cp is None else cp

        mesh = self.mesh

        face = mesh.entity('face')
        fm = self.fm

        face2cell = mesh.face_to_cell()
        qf = mesh.quadrature_formula(p+3, etype='face', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, NFV)  (NQ,)
        ps = self.mesh.face_bc_to_point(bcs) # (NF, NQ, 3)

        phi0 = self.face_basis(ps, p=p) # (NF, NQ, fldof)
        phi1 = self.basis(ps, index=face2cell[:, 0], p=cp) # (NF, NQ, cldof)
        phi2 = self.basis(ps, index=face2cell[:, 1], p=cp) # (NF, NQ, cldof)
        LM = bm.einsum('j, ijk, ijm, i->ikm', ws, phi0, phi1, fm)
        RM = bm.einsum('j, ijk, ijm, i->ikm', ws, phi0, phi2, fm)
        return LM, RM
    
    def cell_hessian_matrix(self, p=None):
        """
        Cell hessian matrix, shape:(NC, ldof, ldof)
        This program is only applicable to polygon meshes for (\nabla^2 u, \nabla^2 v).
        Parameters:
            p : int, optional
                The degree of the polynomial space. If None, use the default degree of the space.
        Returns:
            A : The cell hessian matrix. shape is (NC, ldof, ldof)
        """
        p = self.p if p is None else p
        
        @cartesian
        def f(x, index):
            hphi = self.hessian_basis(x, index=index, p=p)
            return bm.einsum('cqlij, cqmij->cqlm', hphi, hphi)
        A = self.mesh.integral(f, q=p+3, celltype=True) # (NC, ldof, ldof)
        return A
    
    @cartesian
    def to_cspace_function(self, uh):
        """
        Restore the function uh from the piecewise p-th degree polynomial space
        to a piecewise continuous function space. This assumes a tetrahedral mesh.

        Parameters
            uh : TensorLike
                The function in the piecewise p-th degree polynomial space.
        Returns
            TensorLike
                The function in the piecewise continuous function space.
        TODO
        ----
        1. 实现多个函数同时恢复的情形 
        """
        from ..functionspace import LagrangeFESpace
        # number of function in uh

        p = self.p
        mesh  = self.mesh
        bcs = bm.multi_index_matrix(p, dim=3)
        ps = mesh.bc_to_point(bcs)
        val = self.value(uh, ps) # （NQ, NC, ...)

        space = LagrangeFESpace(mesh, p=p)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        deg = bm.zeros(gdof, **self.fkwargs)
        bm.index_add(deg, cell2dof, 1)
        ruh = space.function()
        bm.index_add(ruh, cell2dof, val.T)
        ruh /= deg
        return ruh
    
    def show_frame(self, axes, index=1):
        n = bm.array([[1.0, 2.0, 1.0], [-1.0, 2.0, 1.0]], **self.fkwargs)/bm.sqrt(6)
        a, b, frame = bm.linalg.svd(n[:, None, :])
        a = a.reshape(-1)
        frame[a == 1, 2, :] *= -1
        frame[a ==-1] *=-1

        c = ['r', 'g', 'b']
        for i in range(3):
            axes.quiver(
                    0.0, 0.0, 0.0, 
                    frame[index, i, 0], frame[index, i, 1], frame[index, i, 2],
                    length=0.1, normalize=True, color=c[i])

    def show_cell_basis_index(self, p=1):
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d as a3
        from scipy.spatial import Delaunay
        import sympy as sp
        from sympy.abc import x, y, z

        from ..mesh import TetrahedronMesh

        index = bm.multi_index_matrix(p,3)
        phi = x**index[:, 1]*y**index[:, 2]*z**index[:, 3]
        phi = ['$'+x+'$' for x in map(sp.latex, phi)]
        bc = index/p

        mesh0 = TetrahedronMesh.from_one_tetrahedron()
        node0 = mesh0.entity('node')

        # plot
        fig = plt.figure()
        axes = fig.add_subplot(131, projection='3d')
        axes.set_axis_off()

        edge0 = bm.array([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], **self.ikwargs)
        lines = a3.art3d.Line3DCollection(node0[edge0], color='k', linewidths=2)
        axes.add_collection3d(lines)

        edge1 = bm.array([(0, 2)], **self.ikwargs)
        lines = a3.art3d.Line3DCollection(node0[edge1], color='gray', linewidths=2,
                alpha=0.5)
        axes.add_collection3d(lines)
        mesh0.find_node(axes, showindex=True, color='k', fontsize=15,
                markersize=10)

        node1 = mesh0.bc_to_point(bc).reshape(-1, 3)
        idx = bm.arange(1, p+2 , **self.ikwargs)
        idx = bm.cumsum(bm.cumsum(idx))

        d = Delaunay(node1)
        mesh1 = TetrahedronMesh(node1, d.simplices)

        face = mesh1.entity('face')
        isFace = bm.zeros(len(face), dtype=bm.bool)
        for i in range(len(idx)-1):
            flag = bm.sum((face >= idx[i]) & (face < idx[i+1]), axis=-1) == 3
            isFace[flag] = True
        face = face[isFace]

        axes = fig.add_subplot(132, projection='3d')
        axes.set_axis_off()

        lines = a3.art3d.Line3DCollection(node0[edge0], color='k', linewidths=2)
        axes.add_collection3d(lines)

        lines = a3.art3d.Line3DCollection(node0[edge1], color='gray', linewidths=2,
                alpha=0.5)
        axes.add_collection3d(lines)
        faces = a3.art3d.Poly3DCollection(node1[face], facecolor='w', edgecolor='k',
                linewidths=1, linestyle=':', alpha=0.3)
        axes.add_collection3d(faces)
        mesh1.find_node(axes, showindex=True, color='r', fontsize=15,
                markersize=10)

        axes = fig.add_subplot(133, projection='3d')
        axes.set_axis_off()
        lines = a3.art3d.Line3DCollection(node0[edge0], color='k', linewidths=2)
        axes.add_collection3d(lines)

        lines = a3.art3d.Line3DCollection(node0[edge1], color='gray', linewidths=2,
                alpha=0.5)
        axes.add_collection3d(lines)
        faces = a3.art3d.Poly3DCollection(node1[face], facecolor='w', edgecolor='k',
                linewidths=1, linestyle=':', alpha=0.3)
        axes.add_collection3d(faces)
        mesh1.find_node(axes, showindex=True, color='r', fontsize=15,
                markersize=10, multiindex=phi)

        plt.show()