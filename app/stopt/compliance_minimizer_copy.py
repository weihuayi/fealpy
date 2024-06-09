import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from utilfuncs import Mesher

@jit
# 投影滤波器
def projectionFilter(projection, rho):
    if(projection['isOn']):
        v1 = jnp.tanh(projection['c0'] * projection['beta'])
        nm = v1 + jnp.tanh(projection['beta'] * (rho - projection['c0']))
        dnm = v1 + jnp.tanh(projection['beta'] * (1. - projection['c0']))
        return nm/dnm
    else:
        return rho
    
@jit
# SIMP 材料插值模型
def materialModel(material, rho):
    E = material['Emin'] + \
        (material['Emax'] - material['Emin']) * \
        (rho + 0.01) ** material['penal']
    return E

@jit
# 组装全局刚度矩阵
def assembleK(mesh, K0, E, idx):
    K_asm = jnp.zeros((mesh['ndof'], mesh['ndof']))
    K_elem = (K0.flatten()[jnp.newaxis]).T

    K_elem = (K_elem * E).T.flatten()
    K_asm = K_asm.at[(idx)].add(K_elem)
    return K_asm

@jit
# 直接法求解线性方程组
def solveKuf(bc, K, mesh):
    u_free = jax.scipy.linalg.solve(K[bc['free'],:][:, bc['free']], \
                                    bc['force'][bc['free']], check_finite=False)
    u = jnp.zeros((mesh['ndof']))
    u = u.at[bc['free']].set(u_free.reshape(-1))
    return u

class ComplianceMinimizer:
    def __init__(self,
                 mesh = None,
                 bc = None,
                 material = None,
                 globalvolCons = None,
                 optimizationParams = None,
                 projection = None):
        
        # 默认网格参数
        if mesh is None:
            nelx, nely = 60, 30
            elemSize = jnp.array([1., 1.])
            mesh = {'nelx': nelx, 'nely': nely, 'elemSize': elemSize,
                    'ndof': 2 * (nelx + 1) * (nely + 1), 'numElems': nelx * nely}
        else:
            nelx, nely = mesh['nelx'], mesh['nely']
        
        # 默认材料参数
        if material is None:
            material = {'Emax': 1., 'Emin': 1e-3, 'nu': 0.3, 'penal': 3.}
        
        # 默认全局体积约束
        if globalvolCons is None:
            globalvolCons = {'isOn': True, 'vf': 0.5}
        
        # 默认优化器参数
        if optimizationParams is None:
            optimizationParams = {'maxIters': 200, 'minIters': 100, 'relTol': 0.05}

        # 默认投影滤波器参数
        if projection is None:
            projection = {'isOn': True, 'beta': 4, 'c0': 0.5}
        
        # 默认边界条件和载荷
        if bc is None:
            example = 1
            if example == 1:
                force = jnp.zeros((mesh['ndof'], 1))
                dofs = jnp.arange(mesh['ndof'])
                fixed = dofs[0:2 * (nely + 1):1]
                free = jnp.setdiff1d(jnp.arange(mesh['ndof']), fixed)
                # JAX 数组是不可变的，不能直接进行赋值操作。需要使用 JAX 提供的 .at[].set() 方法来进行修改
                force = force.at[2 * (nelx + 1) * (nely + 1) - 2 * nely + 1, 0].set(-1)
                symXAxis = False
                symYAxis = False
            elif example == 2:
                force = jnp.zeros((mesh['ndof'], 1))
                dofs = jnp.arange(mesh['ndof'])
                fixed = dofs[0:2 * (nely + 1):1]
                free = jnp.setdiff1d(jnp.arange(mesh['ndof']), fixed)
                # JAX 数组是不可变的，不能直接进行赋值操作。需要使用 JAX 提供的 .at[].set() 方法来进行修改
                force = force.at[2 * (nelx + 1) * (nely + 1) - (nely + 1), 0].set(-1)
                symXAxis = True
                symYAxis = False
            else:
                force = jnp.zeros((mesh['ndof'], 1))
                fixed = jnp.array([])
                free = jnp.array([])
                symXAxis = False
                symYAxis = False

            bc = {'force': force, 'fixed': fixed, 'free': free,
                  'symXAxis': symXAxis, 'symYAxis': symYAxis}

        self.mesh = mesh
        self.material = material
        self.bc = bc
        
        # 初始化 Mesher 类并获取初始刚度矩阵
        M = Mesher()
        self.edofMat, self.idx = M.getMeshStructure(mesh)
        self.K0 = M.getK0(self.material)

        # 设置全局体积约束
        self.globalVolumeConstraint = globalvolCons
        
        # 自动微分计算柔顺度及其灵敏度
        self.objectiveHandle = value_and_grad(self.computeCompliance)
        
        # 自动微分计算约束值及其灵敏度
        self.consHandle = self.computeConstraints
        self.numConstraints = 1
        
        # 设置优化器参数
        self.optimizationParams = optimizationParams

        # 设置投影滤波器参数
        self.projection = projection

    def projectionFilter(self, rho):
        return projectionFilter(self.projection, rho)
    
    def materialModel(self, rho):
        return materialModel(self.material, rho)
    
    def assembleK(self, E):
        return assembleK(self.mesh, self.K0, E, self.idx)
    
    def solveKuf(self, K):
        return solveKuf(self.bc, K, self.mesh)   

    def computeCompliance(self, rho):
        
        rho = self.projectionFilter(rho)
        E = self.materialModel(rho)
        K = self.assembleK(E)
        u = self.solveKuf(K)
        J = jnp.dot(self.bc['force'].T, u)[0]

        return J
    
    def computeConstraints(self, rho, epoch):

        @jit
        # 计算体积约束
        def computeGlobalVolumeConstraint(rho):
            g = jnp.mean(rho) / self.globalVolumeConstraint['vf'] - 1.
            return g
        
        # 体积约束的值及其灵敏度
        c, gradc = value_and_grad(computeGlobalVolumeConstraint)(rho)
        c, gradc = c.reshape((1, 1)), gradc.reshape((1, -1))

        return c, gradc

    def mmaOptimize(self, optimizationParams, ft):
        
        rho = jnp.ones((self.mesh['nelx'] * self.mesh['nely']))
        loop = 0
        change = 1.

        while( (change > optimizationParams['relTol']) \
            and (loop < optimizationParams['maxIters']) \
            or (loop < optimizationParams['minIters']) ):

            loop = loop + 1

            J, dJ = self.objectiveHandle(rho)

            vc, dvc = self.consHandle(rho, loop)

    


