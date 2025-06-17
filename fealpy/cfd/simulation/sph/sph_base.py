from fealpy.backend import backend_manager as bm
from .particle_kernel_function_new import QuinticKernel, CubicSplineKernel, QuadraticKernel, WendlandC2Kernel, QuinticWendlandKernel
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

    @classmethod
    def create(cls, kinfo):
        """根据参数字典创建核函数实例"""
        ktype = kinfo.get("type", "quintic")
        h = kinfo.get("h")
        if ktype not in cls._registry:
            raise ValueError(f"Unknown kernel type: {ktype}")
        return cls._registry[ktype](h)

    def compute_displacement(self, node_self, neighbors, r, box_size):
        """计算位移差向量"""
        if self.use_space and self.space is not None:
            displacement, shift = self.space.periodic(side=box_size)
            r_i_s, r_j_s = r[neighbors], r[node_self]
            dr = bm.vmap(displacement)(r_i_s, r_j_s)
        else:
            r_i_s, r_j_s = r[node_self], r[neighbors]
            dr = r_i_s - r_j_s
        return dr

    def compute_distance(self, node_self, neighbors, r, box_size):
        """计算节点间的距离"""
        dr = self.compute_displacement(node_self, neighbors, r, box_size)
        
        if self.use_space and self.space is not None:
            return self.space.distance(dr)
        else:
            return bm.linalg.norm(dr, axis=1)

    def compute_kernel_value(self, node_self, neighbors, r, box_size):
        """计算核函数值"""
        dist = self.compute_distance(node_self, neighbors, r, box_size)
        return bm.vmap(self.kernel_func.value)(dist)

    def compute_kernel_gradient(self, node_self, neighbors, r, box_size):
        """计算核函数梯度"""
        dist = self.compute_distance(node_self, neighbors, r, box_size)
        dr = self.compute_displacement(node_self, neighbors, r, box_size)
        
        EPS = bm.finfo(float).eps
        e = dr / (dist[:, None] + EPS)  # 单位向量
        grad_norm = bm.vmap(self.kernel_func.grad_value)(dist)
        grad = grad_norm[:, None] * e
        return grad, grad_norm

class SPHQueryKernel:
    def __init__(self, mesh, radius=None, box_size=None, mask_self=True, kernel_info=None, periodic=[False, False, False]):
        """
        参数:
            mesh: 网格对象
            radius: 邻近搜索半径，默认为3*dx
            box_size: 模拟区域尺寸，默认为坐标最大值
            kernel_info: 核函数参数字典，默认为quintic核
            periodic: 周期性边界条件，默认为[False, False, False]
            mask_self: 是否屏蔽自身节点，默认为True
        """
        if mesh is None:
            raise ValueError("Mesh object must be provided.")
        
        self.mesh = mesh
        self.dx = mesh.nodedata["dx"]
        self.pos = mesh.nodedata["position"]
        self.dim = self.pos.shape[1]
        self.radius = radius if radius is not None else 3 * self.dx
        self.box_size = box_size if box_size is not None else self.pos.max(axis=0)
        self.periodic = periodic
        self.mask_self = mask_self

        # 初始化核函数
        if kernel_info is None:
            kernel_info = {'type': 'quintic', 'h': self.dx, 'space': True}
        self.kernel = Kernel(kernel_info, dim=self.dim)

    def find_node(self, other_pos=None, device=None):
        """执行邻近搜索，返回 node_self 和 neighbors"""
        pos_cpu = self.pos
        other_cpu = other_pos if other_pos is not None else self.pos

        # 保存原始 GPU 数据
        pos_gpu = self.pos
        other_gpu = other_pos

        if device == 'gpu':
            # 转为 CPU 数据类型
            pos_cpu = pos_gpu.cpu().numpy() if hasattr(pos_gpu, "cpu") else pos_gpu
            other_cpu = other_gpu.cpu().numpy() if (other_gpu is not None and hasattr(other_gpu, "cpu")) else other_cpu

        elif device not in {None, 'cpu'}:
            raise NotImplementedError("Unsupported device: only 'cpu' and 'gpu' are supported.")

        # 执行邻近搜索
        node_self, neighbors = bm.query_point(pos_cpu, other_cpu, self.radius, self.box_size, self.mask_self, self.periodic)

        # 恢复 self.pos 和 other_pos 为 GPU
        if device == 'gpu':
            self.pos = pos_gpu
            if other_pos is not None:
                other_pos = other_gpu

        return node_self, neighbors

    def compute_kernel_value(self, node_self, neighbors):
        """计算核函数值"""
        return self.kernel.compute_kernel_value(node_self, neighbors, self.pos, self.box_size)

    def compute_kernel_gradient(self, node_self, neighbors):
        """计算核函数梯度"""
        return self.kernel.compute_kernel_gradient(node_self, neighbors, self.pos, self.box_size)