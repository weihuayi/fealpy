
from ..backend import backend_manager as bm
from ..functionspace import LagrangeFESpace
from ..functionspace import TensorFunctionSpace
from ..fem import BilinearForm, LinearElasticIntegrator
from ..fem import LinearForm, VectorSourceIntegrator
from ..fem import DirichletBC



class LinearElasticityLFEMSolver:
    """
    采用 Lagrange 有限元求解线弹性方程
    """

    def __init__(self, material, mesh, p, method=None):
        """
        生成线弹性离散系统
        """
        assert p >= 1, f"p={p} should be greater than 1!"
        self.mesh = mesh
        self.material = material

        self.vspace = TensorFunctionSpace(
                LagrangeFESpace(mesh, p=p), 
                (-1, mesh.geo_dimension())
                ) 

        self.bform = BilinearForm(self.vspace)
        self.bform.add_integrator(
                LinearElasticIntegrator(material, method=None)
                )
        self.A = self.bform.assembly()

        self.lform = LinearForm(self.vspace)
        self.b = self.lform.assembly()

    def apply_node_load(self):
        """
        """

        mesh = self.mesh
        node = mesh.entity('node')
        kwargs = bm.context(node)
        GD = mesh.geo_dimension()
        NN = mesh.number_of_nodes()
        b = bm.zeros((NN, GD), **kwargs)
        
        for name in mesh.meshdata['load']['node']:
            nload = self.mesh.meshdata['load']['node'][name]
            idx = nload['nset']
            b = bm.set_at(b, (idx, slice(GD)), nload['value'])
            self.b = self.b + b.reshape(-1)

    def apply_face_load(self):
       
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        GD = mesh.geo_dimension()
        node = mesh.entity('node')
        kwargs = bm.context(node)
        f = bm.zeros((NN, GD), **kwargs)
        for name in self.mesh.meshdata['load']['face']:
            fload = self.mesh.meshdata['load']['face'][name]
            if 'nset' in fload: 
                idx = fload['nset']
                f = bm.set_at(f, (idx, slice(GD)), fload['value'])
                print(f[idx])
                f = self.vspace.function(array=f.reshape(-1))



    def apply_dirichlet_bc(self, threshold):
        self.A, self.b = DirichletBC(self.vspace, gd=0.0, 
                                     threshold=threshold).apply(self.A, self.b)

    def set_corner_disp_zero(self):
        """
        修改线性系统，使得
        """
        from ..fem import DirichletBC
        mesh = self.vspace.mesh
        gdof = self.vspace.number_of_global_dofs() 
        assert 'cornerIdx' in mesh.meshdata, f"mesh.meshdata does't exist cornerIdx!"
        cornerIdx = mesh.meshdata['cornerIdx']
        gdof = self.vspace.number_of_global_dofs()
        isCornerDof = bm.zeros(gdof, dtype=bm.bool) 
        isCornerDof = bm.set_at(isCornerDof, 2*cornerIdx, True)
        isCornerDof = bm.set_at(isCornerDof, 2*cornerIdx + 1, True)

        bc = DirichletBC(self.vspace, gd=0.0, threshold=isCornerDof)
        self.A, self.b = bc.apply(self.A, self.b)

    def set_normal_disp_zero(self, domain):  
        """
        扩展系统，利用乘子法，法向位移设置为0
        """
        assert bm.backend_name == 'numpy', f"only work for numpy backend, the current backend is {bm.backend_name}!" 

        from scipy.sparse import csr_matrix, bmat
        from scipy.sparse.linalg import spsolve

        gdof = self.vspace.number_of_global_dofs() 
        mesh = self.vspace.mesh
        NN = mesh.number_of_nodes()
        node = mesh.entity('node')
        bdNodeIdx = mesh.boundary_node_index()

        kargs = bm.context(node)
        kargs['dtype'] = bm.bool
        isBdNode = bm.zeros(NN, **kargs)
        isBdNode = bm.set_at(isBdNode, bdNodeIdx, True)
        cornerIdx = mesh.meshdata['cornerIdx']
        isBdNode = bm.set_at(isBdNode, cornerIdx, False)



        bdNode = node[isBdNode]
        nd = bdNode.shape[0] # 非角点边界节点数
        n = domain.grad_signed_dist_function(bdNode) # 非角点边界节点的法向量
        val = n.reshape(-1) # 非角点边界节点的法向量
        I = bm.repeat(range(nd), 2).reshape(-1)
        bdNodeIdx, = bm.where(isBdNode)
        J = bm.stack([2*bdNodeIdx, 2*bdNodeIdx+1], axis=1).reshape(-1)

        G = csr_matrix((val, (I, J)), shape=(nd, gdof))
        self.A = bmat([[self.A.to_scipy(), G.T], [G, None]], format='csr')

        kargs = bm.context(node)
        self.b = bm.concatenate([self.b, bm.zeros(nd, **kargs)], axis=0)


    def solve(self):
        """
        """
        from ..solver.mumps import spsolve
        mesh = self.mesh
        GD = mesh.geo_dimension()
        gdof = self.vspace.number_of_global_dofs() 
        du = spsolve(self.A, self.b)
        self.du = du[:gdof].reshape(-1, GD)
        return self.du

    def show_mesh(self, **kwargs):
        """
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        mesh = self.mesh
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        #mesh.add_plot(axes)
        mesh.find_node(axes, index=kwargs['nindex'], showindex=True)
        plt.show()

    def show_displacement(self, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        mesh = self.mesh
        node = mesh.entity('node')
        node += kwargs['alpha']*self.du
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        mesh.add_plot(axes)
        #axes.quiver(node[:, 0], node[:, 1], node[:, 2], self.du[:, 0], self.du[:, 1], self.du[:, 2])
        plt.show()
