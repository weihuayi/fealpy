from fealpy.backend import backend_manager as bm
from .sph.particle_kernel_function_new import QuinticKernel, CubicSplineKernel, QuadraticKernel, WendlandC2Kernel, QuinticWendlandKernel
from fealpy.mesh.node_mesh import Space

class SPHParameters:
    """全局光滑粒子流体动力学默认参数基类"""
    def __init__(self, mesh, **kwargs):
        if mesh is None:
            raise ValueError("Mesh object must be provided.")
        self.mesh = mesh
        self.dim = mesh.nodedata["position"].shape[1]
        self.dx = mesh.nodedata["dx"]

        _defaults = {
        'kernel':{'type': 'quintic', 'h': self.dx, 'space': False},
        'grad_kernel':{'type': 'quintic', 'h': self.dx, 'space': False},
    }

        self._params = self._defaults.copy()
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict) and key in self._params:
                # 更新字典类型的参数
                self._params[key].update(value)
            else:
                # 直接赋值
                self._params[key] = value

    def __getattr__(self, name):
        """通过属性访问参数"""
        if name in self._params:
            return self._params[name]
        raise AttributeError(f"参数 '{name}' 不存在")

    def __getitem__(self, key):
        return self._params[key]

    def __str__(self):
        """自定义打印输出"""
        lines = ["=== SPHParameters ==="]
        for key, value in self._params.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        """调试用表示"""
        return f"SPHParameters(params={self._params})"

class Kernel:
    """核函数注册与创建工具类"""
    _registry = {
        'quintic': QuinticKernel,
        'cubic': CubicSplineKernel,
        'quadratic': QuadraticKernel,
        'wendlandc2': WendlandC2Kernel,
        'quintic_wendland': QuinticWendlandKernel,
    }

    def __init__(self, kinfo, dim, space=None):
        """初始化核函数"""
        self.ktype = kinfo.get("type", "quintic")
        self.h = kinfo.get("h")
        self.dim = dim
        self.use_space = kinfo.get("space", False)
        self.space = Space()
        if self.ktype not in self._registry:
            raise ValueError(f"未知的核函数类型: {self.ktype}")
        self.kernel_func = self._registry[self.ktype](self.h, self.dim)

    def compute_kernel(self, node_self, neighbors, r, box_size):
        """
        计算核函数值和梯度
        参数:
            node_self: 自身节点的索引
            neighbors: 邻居节点的索引
            r: 所有节点的坐标
            box_size: 模拟区域的尺寸
        返回:
            w_dist: 核函数值
            grad_w_dist: 核函数梯度
        """
        EPS = bm.finfo(float).eps

        if self.use_space and self.space is not None:
            # 当 space=True 时，使用周期边界计算距离
            displacement, shift = self.space.periodic(side=box_size)

            r_i_s, r_j_s = r[neighbors], r[node_self]
            dr_i_j = bm.vmap(displacement)(r_i_s, r_j_s)
            dist = self.space.distance(dr_i_j)
            w_dist = bm.vmap(self.kernel_func.value)(dist)

            e_s = dr_i_j / (dist[:, None] + EPS)  # 单位向量 (dr/dx, dr/dy)
            grad_w_dist_norm = bm.vmap(self.kernel_func.grad_value)(dist)
            grad_w_dist = grad_w_dist_norm[:, None] * e_s
        else:
            # 当 space=False 时，使用欧几里得距离
            r_i_s, r_j_s = r[node_self], r[neighbors]
            dr_i_j = r_i_s - r_j_s
            dist = bm.linalg.norm(dr_i_j, axis=1)
            w_dist = bm.vmap(self.kernel_func.value)(dist)

            e_s = dr_i_j / (dist[:, None] + EPS)
            grad_w_dist_norm = bm.vmap(self.kernel_func.grad_value)(dist)
            grad_w_dist = grad_w_dist_norm[:, None] * e_s

        return w_dist, grad_w_dist, dr_i_j, dist, grad_w_dist_norm

    @classmethod
    def create(cls, kinfo):
        """根据参数字典创建核函数实例"""
        ktype = kinfo.get("type", "quintic")
        h = kinfo.get("h")
        if ktype not in cls._registry:
            raise ValueError(f"Unknown kernel type: {ktype}")
        return cls._registry[ktype](h)

class QueryPoint:
    """封装的邻近搜索功能类"""
    def __init__(self, mesh, radius=None, box_size=None, mask_self=True, periodic=[False, False, False]):
        if mesh is None:
            raise ValueError("Mesh must be provided")
    
        self.mesh = mesh
        self.pos = mesh.nodedata["position"]
        self.dim = self.pos.shape[1]
        self.dx = mesh.nodedata["dx"]

        self.radius = radius if radius is not None else 3 * self.dx
        self.box_size = box_size if box_size is not None else self.pos.max(axis=0)
        #self.periodic = self._check_periodic(periodic)
        self.periodic = periodic
        self.mask_self = mask_self

    def query_point(self, other_pos=None):
        """执行邻近搜索，返回 node_self 和 neighbors"""
        if other_pos is None:
            other_pos = self.pos
        return bm.query_point(self.pos, other_pos, self.radius, self.box_size, self.mask_self, self.periodic)



class SPHQueryKernel:
    """整合邻近搜索和核函数计算的功能类"""
    def __init__(self, mesh, radius=None, box_size=None, kernel_info=None, periodic=[False, False, False]):
        """
        初始化邻近搜索和核函数计算
        参数:
            mesh: 网格对象
            radius: 邻近搜索半径，默认为3*dx
            box_size: 模拟区域尺寸，默认为坐标最大值
            kernel_info: 核函数参数字典，默认为quintic核
            periodic: 周期性边界条件，默认为[False, False, False]
        """
        if mesh is None:
            raise ValueError("Mesh object must be provided.")
        
        self.mesh = mesh
        self.dx = mesh.nodedata["dx"]
        self.pos = mesh.nodedata["position"]
        self.dim = self.pos.shape[1]

        # 初始化邻近搜索
        self.query = QueryPoint(
            mesh=mesh,
            radius=radius if radius is not None else 3 * self.dx,
            box_size=box_size if box_size is not None else self.pos.max(axis=0),
            mask_self=True,
            periodic=periodic
        )

        # 初始化核函数
        if kernel_info is None:
            kernel_info = {'type': 'quintic', 'h': self.dx, 'space': True}
        self.kernel = Kernel(kernel_info, dim=self.dim)

    def compute(self):
        """
        执行邻近搜索并计算核函数值和梯度
        返回:
            node_self: 自身节点索引
            neighbors: 邻居节点索引
            w_dist: 核函数值
            grad_w_dist: 核函数梯度
        """
        # 执行邻近搜索
        node_self, neighbors = self.query.query_point()

        # 计算核函数值和梯度
        w_dist, grad_w_dist, dr_i_j, dist, dw_norm = self.kernel.compute_kernel(
            node_self, neighbors, self.pos, self.query.box_size
        )

        return node_self, neighbors, w_dist, grad_w_dist, dr_i_j, dist, dw_norm
