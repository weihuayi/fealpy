from typing import Union, Dict
from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
from fealpy.sparse import COOTensor
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, FunctionSpace

class FEMParameters:
    """全局有限元默认参数基类"""
    _defaults = {
        'uspace': {'type': 'Lagrange','p': 2},
        'pspace': {'type': 'Lagrange','p': 1},
        'tspace': {'type': 'Lagrange','p': 2},
        'assembly': {
            'quadrature_order': None  # 积分阶数
            }
    }
    
    def __init__(self, **kwargs):
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
        lines = ["=== FEMParameters ==="]
        for key, value in self._params.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def __repr__(self):
        """调试用表示"""
        return f"FEMParameters(params={self._params})"


class FEM:
    def __init__(self, equation, params: FEMParameters = None):
        self.equation = equation
        self.mesh = equation.pde.mesh
        self.params = params if params else FEMParameters()         
        self.set = self.Set(self)
        
        self._pspace = self._create_space('pspace')
        uspace = self._create_space('uspace')
        self._uspace = TensorFunctionSpace(uspace, (self.mesh.GD,-1))
        self._tspace = self._create_space('tspace') 
        
        self._q = self.params.assembly['quadrature_order']

    
    def update_mesh(self, mesh):
        self.mesh = mesh
        uspace = self._create_space('uspace')
        self._uspace = TensorFunctionSpace(uspace, (self.mesh.GD,-1))
        self._pspace = self._create_space('pspace')
        self._tspace = self._create_space('tspace')

    @property
    def uspace(self) -> FunctionSpace:
        return self._uspace
    @property
    def pspace(self) -> FunctionSpace:
        return self._pspace
    @property
    def tspace(self) -> FunctionSpace:
        return self._tspace
    @property
    def q(self) -> int:
        return self._q

    def _create_space(self, space_name: str) -> FunctionSpace:
        """根据配置动态创建空间"""
        config = self.params[space_name]

        # 动态创建空间
        if config['type'] == 'Lagrange':
            return LagrangeFESpace(self.mesh, p=config['p'])
        # 其他空间类型...
        raise ValueError(f"不支持的空间类型: {space_type}")  
    
    def lagrange_multiplier(self, A, b):

        LagLinearForm = LinearForm(self.pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()
        LagA = bm.concatenate([bm.zeros(self.uspace.number_of_global_dofs()), LagA], axis=0)

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                 bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b  = bm.concatenate([b, b0], axis=0)
        return A, b

    class Set:
        """参数设置子类（同步修改参数和空间）"""
        def __init__(self, parent):
            self._parent = parent

        def _update_space(self, space_name: str,
                        space: Union[str, FunctionSpace] = None,
                        p: int = None,
                        **kwargs):
            """通用空间更新方法"""
            # 更新参数
            updates = {}
            if isinstance(space, str):
                updates['type'] = space
            elif isinstance(space, FunctionSpace):
                if isinstance(space, TensorFunctionSpace):
                    updates['type'] = space.scalar_space.__str__().split()[0]
                    updates['p'] = space.scalar_space.p
                else:
                    updates['type'] = space.__str__().split()[0]
                    updates['p'] = space.p

            if p is not None:
                updates['p'] = p
            updates.update(kwargs)

            self._parent.params.update(**{space_name: updates})
            
            property_name = '_' + space_name
            # 重新创建空间（会自动更新__str__）
            setattr(self._parent, property_name, self._parent._create_space(space_name))
            return self
        

        def uspace(self, space: Union[str, FunctionSpace] = None,
                  p: int = None, **kwargs) -> 'FEM.Set':
            updates = {}
            control = False
            if isinstance(space, str):
                updates['type'] = space
            elif isinstance(space, FunctionSpace):
                if not hasattr(space, 'scalar_space'):
                    raise AttributeError(f"'{space}' must be a tensor space")
                updates['type'] = space.scalar_space.__str__().split()[0]
                updates['p'] = space.scalar_space.p
                control = True
            if p is not None:
                updates['p'] = p
            self._parent.params.update(**{'uspace': updates})
            if control:
                self._parent._uspace = space
            else:
                sspace = self._parent._create_space('uspace')
                self._parent._uspace = TensorFunctionSpace(sspace, (self._parent.mesh.GD,-1))
            return self

        def pspace(self,
                  space: Union[str, FunctionSpace] = None,
                  p: int = None) -> 'FEM.Set':
            return self._update_space('pspace', space, p)

        def tspace(self,
                  space: Union[str, FunctionSpace] = None,
                  p: int = None) -> 'FEM.Set':
            return self._update_space('tspace', space, p)

        def assembly(self,
                   quadrature_order: int = None) -> 'FEM.Set':
            updates = {}
            if quadrature_order is not None:
                updates['quadrature_order'] = quadrature_order
            self._parent.params.update(assembly=updates)
            self._parent._q = self._parent.params.assembly['quadrature_order']
            return self



