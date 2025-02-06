
from ..backend import backend_manager as bm
from ..functionspace import TensorFunctionSpace
from ..fem import BilinearForm, LinearElasticIntegrator
from ..fem import LinearForm, VectorSourceIntegrator



class LinearElasticityLFEMSolver:
    """
    采用 Lagrange 有限元求解线弹性方程
    """

    def __init__(self, space, GD, material, force):
        """
        生成线弹性离散系统
        """
        self.vspace = TensorFunctionSpace(space, (-1, GD)) 
        self.material = material
        self.force = force
        bform = BilinearForm(self.vspace)
        bform.add_integrator(LinearElasticIntegrator(material))
        self.A = bform.assembly()

        lform = LinearForm(self.vspace)
        lform.add_integrator(VectorSourceIntegrator(force))
        self.b = lform.assembly()

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
        from scipy.sparse.linalg import spsolve
        gdof = self.vspace.number_of_global_dofs() 
        du = spsolve(self.A, self.b)
        du = du[:gdof].reshape(-1, 2)
        return du
