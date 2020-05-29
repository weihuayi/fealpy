import numpy as np
from .function import Function
from ..quadrature import PolyhedronMeshIntegralAlg
from ..quadrature import FEMeshIntegralAlg
from ..common import ranges


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
        cdof = self.number_of_local_dofs(p=p, etype='cell')
        cell2dof = np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof

    def face_to_dof(self, p=None):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        fdof = self.number_of_local_dofs(p=p, etype='face')
        face2dof = np.arange(NF*fdof).reshape(NF, fdof)
        return face2dof

    def number_of_local_dofs(self, p=None, etype='cell'):
        p = self.p if p is None else p
        if etype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif etype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif etype in {'edge', 1}:
            return p+1

    def number_of_global_dofs(self, p=None, etype='cell'):
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p, etype=etype)
        return NC*ldof


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

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None, etype='cell'):
        return self.dof.number_of_local_dofs(p=p, etype=etype)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)

    def basis(self, point, index=None, p=None):
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
        cdof = self.number_of_local_dofs(p=p, etype='cell')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(cdof,)
        phi = np.ones(shape, dtype=self.ftype)  # (..., M, ldof)
        index = index if index is not None else np.s_[:] 
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

    def face_basis(self, point, index=None, p=None):
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
        
        fdof = self.number_of_local_dofs(p=p, etype='face')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(fdof,)
        phi = np.ones(shape, dtype=np.float)  # (..., NF, fdof)
        index = index if index is not None else np.s_[:] 
        p2 = (point - self.facebarycenter[index])/h[index].reshape(-1, 1)
        phi[..., 1:3] = np.einsum('...jk, jnk->...jn', p2, frame[:, 1:, :])  
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    def grad_basis(self, point, index=None, p=None):
        pass

    def mass_matrix(self):
        pass

    def cell_mass_matrix(self):
        pass

    def face_mass_matrix(self):
        pass

    def show_frame(self, axes, index=1):
        n = np.array([[1.0, 2.0, 1.0], [-1.0, 2.0, 1.0]], dtype=np.float)/np.sqrt(6)
        a, b, frame = np.linalg.svd(n[:, None, :])
        print(a, a.shape)
        print(frame, frame.shape)
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

        from .femdof import multi_index_matrix3d
        from ..mesh import MeshFactory
        from ..mesh import TetrahedronMesh

        mfactory = MeshFactory()
        bc = multi_index_matrix3d(p)/p

        mesh0 = mfactory.one_tetrahedron_mesh(ttype='equ') # 正四面体
        node0 = mesh0.entity('node')

        # plot
        fig = plt.figure()
        axes = fig.add_subplot(121, projection='3d')
        axes.set_axis_off()

        edge0 = np.array([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int)
        lines = a3.art3d.Line3DCollection(node0[edge0], color='k', linewidths=2)
        axes.add_collection3d(lines)

        edge1 = np.array([(0, 2)], dtype=np.int)
        lines = a3.art3d.Line3DCollection(node0[edge1], color='gray', linewidths=2,
                alpha=0.5)
        axes.add_collection3d(lines)

        mesh0.find_node(axes, showindex=True, color='k', fontsize=15,
                markersize=50)

        node1 = mesh0.bc_to_point(bc).reshape(-1, 3)

        idx = np.arange(1, p+2)
        idx = np.cumsum(np.cumsum(idx))

        d = Delaunay(node1)
        mesh1 = TetrahedronMesh(node1, d.simplices)

        face = mesh1.entity('face')
        isFace = np.zeros(len(face), dtype=np.bool)
        for i in range(len(idx)-1):
            flag = np.sum((face >= idx[i]) & (face < idx[i+1]), axis=-1) == 3
            isFace[flag] = True
        face = face[isFace]

        axes = fig.add_subplot(122, projection='3d')
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
                markersize=50)
        plt.show()

    def show_face_basis_index(self, p=1):

        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from scipy.spatial import Delaunay

        from .femdof import multi_index_matrix2d
        from ..mesh import MeshFactory
        from ..mesh import TriangleMesh

        mfactory = MeshFactory()
        bc = multi_index_matrix2d(p)/p
        mesh0 = mfactory.one_triangle_mesh(ttype='equ') # 正三角形 

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
