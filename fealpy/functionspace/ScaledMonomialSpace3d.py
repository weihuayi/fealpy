import numpy as np
from ..decorator import cartesian
from .Function import Function
from ..quadrature import PolyhedronMeshIntegralAlg
from ..quadrature import FEMeshIntegralAlg
from .LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
from .femdof import multi_index_matrix2d, multi_index_matrix3d


class SMDof3d():
    """
    三维缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cdof = self.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof

    def face_to_dof(self, p=None):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        fdof = self.number_of_local_dofs(p=p, doftype='face')
        face2dof = np.arange(NF*fdof).reshape(NF, fdof)
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


class ScaledMonomialSpace3d():
    def __init__(self, mesh, p, q=None):
        """
        The Scaled Momomial Space in R^3
        
        Parameters
        ----------
        mesh : TetrahedronMesh, PolyhedronMesh or HalfEdgeMesh3d object
        p : the space degree
        q : 


        Note
        ----

        """

        self.p = p
        self.GD = 3
        self.mesh = mesh
        self.dof = SMDof3d(mesh, p)

        self.cellbarycenter = mesh.entity_barycenter('cell')
        self.facebarycenter = mesh.entity_barycenter('face')

        q = q if q is not None else p+3
        mtype = mesh.meshtype
        if mtype in {'polyhedron'}:
            self.integralalg = PolyhedronMeshIntegralAlg(
                    self.mesh, q,
                    cellbarycenter=self.cellbarycenter)
        elif mtype in  {'tet'}:
            self.integralalg = FEMeshIntegralAlg(mesh, q)

        self.integrator = self.integralalg.integrator

        self.cellmeasure = self.integralalg.cellmeasure 
        self.facemeasure = self.integralalg.facemeasure 
        self.edgemeasure = self.integralalg.edgemeasure

        self.cellsize = self.cellmeasure**(1/3)
        self.facesize = self.facemeasure**(1/2)
        self.edgesize = self.edgemeasure

        # get the face frame by svd
        n = mesh.face_unit_normal()
        a, _, self.faceframe = np.linalg.svd(n[:, np.newaxis, :]) 
        # make the frame satisfies right-hand rule and the first vector equal to
        # n
        a = a.reshape(-1)
        self.faceframe[a == 1, 2, :] *= -1
        self.faceframe[a ==-1] *=-1

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
    
    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    def face_to_dof(self, p=None):
        return self.dof.face_to_dof(p=p)

    def function(self, dim=None, array=None, dtype=np.float64):
        f = Function(self, dim=dim, array=array, coordtype='cartesian',
                dtype=dtype)
        return f

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def number_of_local_dofs(self, p=None, doftype='cell'):
        return self.dof.number_of_local_dofs(p=p, doftype=doftype)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)

    def diff_index_1(self, p=None):
        """
        计算关于 x, y, z 分别求一次导数后, 非零的基函数的编号及系数
        Notes
        -----

        """
        p = self.p if p is None else p
        index = multi_index_matrix3d(p)
        
        x, = np.nonzero(index[:, 1] > 0)
        y, = np.nonzero(index[:, 2] > 0)
        z, = np.nonzero(index[:, 3] > 0)

        return {'x':(x, index[x, 1]), 
                'y':(y, index[y, 2]),
                'z':(z, index[z, 3])
                }

    def diff_index_2(self, p=None):
        """
        计算基函数求二阶导数后的非零项
        
        Notes
        -----
        """
        p = self.p if p is None else p
        index = multi_index_matrix3d(p)
        
        xx, = np.nonzero(index[:, 1] > 1)
        yy, = np.nonzero(index[:, 2] > 1)
        zz, = np.nonzero(index[:, 3] > 1)

        xy, = np.nonzero((index[:, 1] > 0) & (index[:, 2] > 0))
        xz, = np.nonzero((index[:, 1] > 0) & (index[:, 3] > 0))
        yz, = np.nonzero((index[:, 2] > 0) & (index[:, 3] > 0))

        return {'xx':(xx, index[xx, 1]*(index[xx, 1]-1)), 
                'yy':(yy, index[yy, 2]*(index[yy, 2]-1)),
                'zz':(zz, index[zz, 3]*(index[zz, 3]-1)),
                'xy':(xy, index[xy, 1]*index[xy, 2]),
                'xz':(xz, index[xz, 1]*index[xz, 3]),
                'yz':(yz, index[yz, 2]*index[yz, 3])
                }

    def face_index_1(self, p=None):
        """
        Parameters
        ----------
        p : >= 1
        """
        p = self.p if p is None else p
        index = multi_index_matrix2d(p)
        x, = np.nonzero(index[:, 0] > 0)
        y, = np.nonzero(index[:, 1] > 0)
        z, = np.nonzero(index[:, 2] > 0)
        return {'x': x, 'y':y, 'z':z}

    @cartesian
    def basis(self, point, index=np.s_[:], p=None):
        """
        Compute the basis values at point in cell

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., NC, 2), NC is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., NC, cdof)

        Notes
        -----

        """
        p = self.p if p is None else p
        h = self.cellsize
        cdof = self.number_of_local_dofs(p=p, doftype='cell')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(cdof,)
        phi = np.ones(shape, dtype=self.ftype)  # (..., M, ldof)
        phi[..., 1:4] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 4
            for i in range(2, p+1):
                n = (i+1)*i//2
                phi[..., start:start+n] = phi[..., start-n:start]*phi[..., [1]]
                phi[..., start+n:start+n+i] = phi[..., start-i:start]*phi[..., [2]]
                phi[..., start+n+i] = phi[..., start-1]*phi[..., 3]
                start += n + i + 1  
        return phi

    @cartesian
    def face_basis(self, point, index=np.s_[:], p=None):
        """
        Compute the basis values at point on each face 

        Parameters
        ----------
        point : ndarray
            The shape of `point` is (..., NF, 3), NC is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., NF, fdof)

        Notes
        -----

        The `faceframe` is local orthogonal coordinate frame system.  
        `faceframe[i, 0, :]` is the fixed unit norm vector of i-th face. 
        `faceframe[i, 1:3, :]` are the two unit tangent vector on i-th face. 

        Dot the 3d vector `point - facebarycenter` with `faceframe[i, 1:, :]`,
        repectively, one can get the local coordinate component on i-th face.

        """
        p = self.p if p is None else p
        h = self.facesize
        bc = self.facebarycenter
        frame = self.faceframe
        
        fdof = self.number_of_local_dofs(p=p, doftype='face')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(fdof,)
        phi = np.ones(shape, dtype=self.ftype)  # (..., NF, fdof)
        p2 = (point - self.facebarycenter[index])/h[index].reshape(-1, 1)
        phi[..., 1:3] = np.einsum('...jk, jnk->...jn', p2, frame[index, 1:, :])  
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    @cartesian
    def edge_basis(self, point, index=np.s_[:], p=None):
        p = self.p if p is None else p
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        ec = self.integralalg.edgebarycenter
        eh = self.integralalg.edgemeasure
        et = self.mesh.edge_unit_tangent()
        val = np.sum((point - ec[index])*et[index], axis=-1)/eh[index]
        phi = np.ones(val.shape + (p+1,), dtype=self.ftype)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., np.newaxis]
            np.multiply.accumulate(phi, axis=-1, out=phi)
        return phi

    @cartesian
    def grad_basis(self, point, index=np.s_[:], p=None):
        p = self.p if p is None else p
        h = self.cellsize
        num = len(h) if type(index) is slice else len(index)
 
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 3)
        phi = self.basis(point, index=index, p=p-1)
        idx = self.diff_index_1(p=p)
        gphi = np.zeros(shape, dtype=np.float)
        x = idx['x']
        y = idx['y']
        z = idx['z']
        gphi[..., x[0], 0] = np.einsum('i, ...i->...i', x[1], phi) 
        gphi[..., y[0], 1] = np.einsum('i, ...i->...i', y[1], phi)
        gphi[..., z[0], 2] = np.einsum('i, ...i->...i', z[1], phi)

        if point.shape[-2] == num:
            return gphi/h[index].reshape(-1, 1, 1)
        elif point.shape[0] == num:
            return gphi/h[index].reshape(-1, 1, 1, 1)

    @cartesian
    def laplace_basis(self, point, index=np.s_[:], p=None):
        p = self.p if p is None else p
        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            xx = idx['xx']
            yy = idx['yy']
            zz = idx['zz']
            lphi[..., xx[0]] += np.einsum('i, ...i->...i', xx[1], phi)
            lphi[..., yy[0]] += np.einsum('i, ...i->...i', yy[1], phi)
        return lphi/area[index].reshape(-1, 1)

    @cartesian
    def hessian_basis(self, point, index=np.s_[:], p=None):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters
        ----------
        point : numpy array
            The shape of point is (..., NC, 2)

        Returns
        -------
        hphi : numpy array
            the shape of hphi is (..., NC, ldof, 2, 2)
        """
        p = self.p if p is None else p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 3, 3)
        hphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.index2(p=p)
            xx = idx['xx']
            yy = idx['yy']
            zz = idx['zz']
            xy = idx['xy']
            xz = idx['xz']
            yz = idx['yz']
            hphi[..., xx[0], 0, 0] = np.einsum('i, ...i->...i', xx[1], phi)
            hphi[..., xy[0], 0, 1] = np.einsum('i, ...i->...i', xy[1], phi)
            hphi[..., xz[0], 0, 2] = np.einsum('i, ...i->...i', xz[1], phi)
            hphi[..., yy[0], 1, 1] = np.einsum('i, ...i->...i', yy[1], phi)
            hphi[..., yz[0], 1, 2] = np.einsum('i, ...i->...i', yz[1], phi)
            hphi[..., zz[0], 2, 2] = np.einsum('i, ...i->...i', zz[1], phi)
            hphi[..., 1, 0] = hphi[..., 0, 1] 
        return hphi/area[index].reshape(-1, 1, 1, 1)

    @cartesian
    def value(self, uh, point, index=np.s_[:]):
        phi = self.basis(point, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        return np.einsum(s1, phi, uh[cell2dof[index]]) #TODO: phi[:, index]?

    @cartesian
    def grad_value(self, uh, point, index=np.s_[:]):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        if (type(index) is np.ndarray) and (index.dtype.name == 'bool'):
            N = np.sum(index)
        elif type(index) is slice:
            N = len(cell2dof)
        else:
            N = len(index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == N:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return np.einsum(s1, gphi, uh[cell2dof[index]]) #TODO: ghpi index
        elif point.shape[0] == N:
            return np.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]]) #TODO: gphi, index

    @cartesian
    def laplace_value(self, uh, point, index=np.s_[:]):
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        return np.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]]) #TODO: lphi, index

    @cartesian
    def hessian_value(self, uh, point, index=None):
        #TODO:
        pass

    def mass_matrix(self, p=None):
        return self.cell_mass_matrix(p=p)

    def cell_mass_matrix(self, p=None):
        """
        """
        b = (self.basis, None, None)
        M = self.integralalg.serial_construct_matrix(b)
        return M 

    def face_mass_matrix(self, p=None):
        p = self.p if p is None else p
        def f(x, index=None):
            phi = self.face_basis(x, index=index, p=p)
            return np.einsum('ijk, ijp->ijkp', phi, phi)
        M = self.integralalg.face_integral(f, q=p+3)
        return M
    
    def face_cell_mass_matrix(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh

        face = mesh.entity('face')
        measure = mesh.entity_measure('face')
        face2cell = mesh.ds.face_to_cell()

        qf = mesh.integrator(p+3, 'face') 
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.bc_to_point(bcs, 'face')
        phi0 = self.face_basis(ps, p=p)
        phi1 = self.basis(ps, index=face2cell[:, 0], p=p+1)
        phi2 = self.basis(ps, index=face2cell[:, 1], p=p+1)

        LM = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi1, measure, optimize=True)
        RM = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi2, measure, optimize=True)
        return LM, RM 

    @cartesian
    def to_cspace_function(self, uh):
        """
        Notes
        -----
        把分片的 p 次多项式空间的函数  uh， 恢复到分片连续的函数空间。这里假定网
        格是四面体网格。

        TODO
        ----
        1. 实现多个函数同时恢复的情形 
        """

        # number of function in uh

        p = self.p
        mesh  = self.mesh
        bcs = multi_index_matrix3d(p)
        ps = mesh.bc_to_point(bcs)
        val = self.value(uh, ps) # （NQ, NC, ...)

        space = LagrangeFiniteElementSpace(mesh, p=p)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        deg = np.zeros(gdof, dtype=space.itype)
        np.add.at(deg, cell2dof, 1)
        ruh = space.function()
        np.add.at(ruh, cell2dof, val.T)
        ruh /= deg
        return ruh

    def show_frame(self, axes, index=1):
        n = np.array([[1.0, 2.0, 1.0], [-1.0, 2.0, 1.0]], dtype=np.float)/np.sqrt(6)
        a, b, frame = np.linalg.svd(n[:, None, :])
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


        from .femdof import multi_index_matrix3d
        from ..mesh import MeshFactory as MF
        from ..mesh import TetrahedronMesh

        index = multi_index_matrix3d(p)
        phi = x**index[:, 1]*y**index[:, 2]*z**index[:, 3]
        phi = ['$'+x+'$' for x in map(sp.latex, phi)]
        bc = index/p

        mesh0 = MF.one_tetrahedron_mesh(meshtype='equ') # 正四面体
        node0 = mesh0.entity('node')

        # plot
        fig = plt.figure()
        axes = fig.add_subplot(131, projection='3d')
        axes.set_axis_off()

        edge0 = np.array([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int)
        lines = a3.art3d.Line3DCollection(node0[edge0], color='k', linewidths=2)
        axes.add_collection3d(lines)

        edge1 = np.array([(0, 2)], dtype=np.int)
        lines = a3.art3d.Line3DCollection(node0[edge1], color='gray', linewidths=2,
                alpha=0.5)
        axes.add_collection3d(lines)
        mesh0.find_node(axes, showindex=True, color='k', fontsize=15,
                markersize=10)


        node1 = mesh0.bc_to_point(bc).reshape(-1, 3)
        idx = np.arange(1, p+2)
        idx = np.cumsum(np.cumsum(idx))

        d = Delaunay(node1)
        mesh1 = TetrahedronMesh(node1, d.simplices)

        face = mesh1.entity('face')
        isFace = np.zeros(len(face), dtype=np.bool_)
        for i in range(len(idx)-1):
            flag = np.sum((face >= idx[i]) & (face < idx[i+1]), axis=-1) == 3
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

    def show_face_basis_index(self, p=1):

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from scipy.spatial import Delaunay

        from .femdof import multi_index_matrix2d
        from ..mesh import MeshFactory
        from ..mesh import TriangleMesh

        bc = multi_index_matrix2d(p)/p
        mesh0 = MF.one_triangle_mesh(ttype='equ') # 正三角形 

        fig = plt.figure()
        axes = fig.add_subplot(121)
        axes.set_axis_off()

        mesh0.add_plot(axes, cellcolor='w', linewidths=2)
        mesh0.find_node(axes, showindex=True, color='k', fontsize=15,
                fontcolor='k', markersize=20)

        axes = fig.add_subplot(122)
        axes.set_axis_off()

        mesh0.add_plot(axes, cellcolor='w', linewidths=2)

        node1 = mesh0.bc_to_point(bc).reshape(-1, 2)
        d = Delaunay(node1)
        mesh1 = TriangleMesh(node1, d.simplices)

        edge1 = mesh1.entity('edge')
        lines = LineCollection(node1[edge1], color='k', linewidths=1, linestyle=':')
        axes.add_collection(lines)

        mesh1.find_node(axes, showindex=True, fontsize=15, markersize=20,
                color='r')
        plt.show()
