import inspect

from ....backend import backend_manager as bm
from ....backend import TensorLike
from ....fem import LinearForm, BilinearForm
from ....fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator, 
                     ViscousWorkIntegrator, SourceIntegrator, GradSourceIntegrator, 
                     BoundaryFaceSourceIntegrator, ScalarDiffusionIntegrator)
from .fem_base import FEM
from .project_method import ProjectionMethod
from ....fem import DirichletBC
from ..simulation_base import SimulationBase, SimulationParameters
from ....functionspace import Function
from ....decorator import barycentric

class IPCS(ProjectionMethod, FEM):
    """IPCS分裂投影法"""
    def __init__(self, equation, boundary_threshold=None):
        FEM.__init__(self, equation)
        ProjectionMethod.__init__(self)
        self.threshold = boundary_threshold
    
    def simulation(self):
        """返回当前的仿真对象"""
        return ipcs_simulation(self)

    def predict_velocity(self, u0:TensorLike, p0:TensorLike, threshold = None, 
                         return_form:bool=True,BC=None):
        if threshold is not None:
            self.threshold = threshold
        
        Bform = self.predict_velocity_BForm()
        Lform = self.predict_velocity_LForm()
        self.predict_velocity_update(u0, p0)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if BC is not None:
                A, b = BC.apply(A, b)
            return A, b 
        else:
            return Bform, Lform 
    
    def pressure(self, us:TensorLike, p0:TensorLike, 
                 return_form=True, BC=None):
        Bform = self.pressure_BForm()
        Lform = self.pressure_LForm()
        self.pressure_update(us, p0)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if BC is not None:
                A, b = BC.apply(A, b)
            return A, b
        else:
            return Bform, Lform 
    
    def correct_velocity(self, us:TensorLike, p0:TensorLike, p1:TensorLike, 
                         return_form=True, BC=None):
        """速度校正"""
        Bform = self.correct_velocity_BForm()
        Lform = self.correct_velocity_LForm()
        self.correct_velocity_update(us, p0, p1)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if BC is not None:
                A, b = BC.apply(A, b)
            return A, b
        else:
            return Bform, Lform 

    def predict_velocity_BForm(self):
        """预测速度左端项"""
        uspace = self.uspace
        q = self.q 
        threshold = self.threshold

        Bform = BilinearForm(uspace)
        self.predict_BM = ScalarMassIntegrator(q=q)
        self.predict_BF = FluidBoundaryFrictionIntegrator(q=q, threshold=threshold)
        self.predict_BVW = ViscousWorkIntegrator(q=q)
        
        Bform.add_integrator(self.predict_BM)
        Bform.add_integrator(self.predict_BF)
        Bform.add_integrator(self.predict_BVW)
        return Bform
    
    def predict_velocity_LForm(self):
        """预测速度右端项"""
        uspace = self.uspace
        q = self.q
        threshold = self.threshold
        
        Lform = LinearForm(uspace) 
        self.predict_LS = SourceIntegrator(q=q)
        self.predict_LS_f = SourceIntegrator(q=q)
        self.predict_LGS = GradSourceIntegrator(q=q)
        self.predict_LBFS = BoundaryFaceSourceIntegrator(q=q, threshold=threshold)
        
        Lform.add_integrator(self.predict_LS)
        Lform.add_integrator(self.predict_LGS)
        Lform.add_integrator(self.predict_LBFS)
        return Lform
    
    def predict_velocity_update(self, u0, p0): 
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative 
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force

        self.predict_BM.coef = ctd/dt
        self.predict_BF.coef = -cv
        self.predict_BVW.coef = 2*cv
        
        @barycentric
        def LS_coef(bcs, index):
            masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = 1/dt*masscoef*u0(bcs, index)
            ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result -= ccoef * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
            result += cbfcoef
            return result
        self.predict_LS.source = LS_coef

        @barycentric
        def LGS_coef(bcs, index):
            I = bm.eye(self.mesh.GD)
            result = bm.repeat(p0(bcs,index)[...,bm.newaxis], self.mesh.GD, axis=-1)
            result = bm.expand_dims(result, axis=-1) * I
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        self.predict_LGS.source = LGS_coef
        
        @barycentric
        def LBFS_coef(bcs, index):
            result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), self.mesh.face_unit_normal(index=index))
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        self.predict_LBFS.source = LBFS_coef
        

    def pressure_BForm(self):
        """压力泊松方程左端项"""
        pspace = self.pspace
        q = self.q
         
        Bform = BilinearForm(pspace)
        self.pressure_BD = ScalarDiffusionIntegrator(q=q)
        Bform.add_integrator(self.pressure_BD) 
        return Bform
    
    def pressure_LForm(self):
        """压力泊松方程右端项"""
        pspace = self.pspace
        q = self.q
         
        Lform = LinearForm(pspace)
        self.pressure_LS = SourceIntegrator(q=q)
        self.pressure_LGS = GradSourceIntegrator(q=q)
        
        Lform.add_integrator(self.pressure_LS)
        Lform.add_integrator(self.pressure_LGS)
        return Lform
    
    def pressure_update(self, us, p0): 
        equation = self.equation
        dt = self.dt
        pc = equation.coef_pressure
        ctd = equation.coef_time_derivative 
        
        self.pressure_BD.coef = pc

        @barycentric
        def LS_coef(bcs, index=None):
            result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
            result *= ctd(bcs, index) if callable(ctd) else ctd
            return result
        self.pressure_LS.source = LS_coef
        
        @barycentric
        def LGS_coef(bcs, index=None):
            result = p0.grad_value(bcs, index)
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        self.pressure_LGS.source = LGS_coef
    
    def correct_velocity_BForm(self):
        """速度校正左端项"""
        uspace = self.uspace
        q = self.q
        
        Bform = BilinearForm(uspace)
        self.correct_BM = ScalarMassIntegrator(q=q)
        Bform.add_integrator(self.correct_BM)
        return Bform
    
    def correct_velocity_LForm(self):
        """速度校正右端项"""
        uspace = self.uspace
        q = self.q
        
        Lform = LinearForm(uspace)
        self.correct_LS = SourceIntegrator(q=q)
        Lform.add_integrator(self.correct_LS)
        return Lform

    def correct_velocity_update(self, us, p0, p1):
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative
        cp = equation.coef_pressure


        self.correct_BM.coef = ctd
        @barycentric
        def BM_coef(bcs, index):
            masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = masscoef * us(bcs, index)
            result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
            return result
        self.correct_LS.source = BM_coef

class ipcs_simulation(SimulationBase):
    def __init__(self, method):
        super().__init__(method)
        self._initialize_variables()
        self.params = IPCSSimulationParameters()

    def _initialize_variables(self):
        """初始化所有未设置的变量空间"""
        equation = self.equation
        for var_name, value in equation._variables.items():
            if value is None:
                equation.set_variable(var_name, self._create_variable_space(var_name).function())
            elif isinstance(value, Function):  
                continue
            elif callable(value):
                equation.set_variable(var_name, self._create_variable_space(var_name).interpolate(value))
            else:
                raise ValueError(f"Unknown variable type: {type(value)}")

    def _create_variable_space(self, var_name):
        """根据变量名创建对应的初始空间"""
        if var_name == 'velocity':
            return self.method.uspace
        elif var_name == 'pressure':
            # 压力场是标量
            return self.method.pspace
        else:
            raise ValueError(f"Unknown variable name: {var_name}")
    
    def solve(self, A, b):
        """根据求解器类型执行求解"""
        solver_type = self.params._params["solver"]["type"]
        if solver_type == "direct":
            from fealpy.solver import spsolve
            return spsolve(A, b, self.params._params["solver"]['params'])
        elif solver_type == "custom":
            solver = self.params._params["solver"]['params']
            return solver(A, b)
        else:
            raise ValueError(f"不支持的求解器类型: {solver_type}")  

    def run(self):
        """运行仿真"""
        t0 = self.params._params["time"]["T0"]
        dt = self.params._params["time"]["dt"]
        nt = self.params._params["time"]["nt"]
        self.method.dt = dt
        equation = self.equation

        # 读取初始条件
        u0 = equation.velocity
        us = u0.space.function()
        u1 = u0.space.function()
        p0 = equation.pressure
        p1 = p0.space.function()
        
        for i in range(nt):
            t = t0 + (i + 1) * dt
            print(f"第 {i+1} 步，时间 = {t}")
            u1, p1 = self.run_one_step(u0, p0)
            u0[:] = u1[:]
            p0[:] = p1[:]
            self.equation.set_variable('velocity', u0)
            self.equation.set_variable('pressure', p0)
            if self.params._params["output"]["onoff"]:
                name = 'test_'+ str(i+1).zfill(10) + '.vtu'
                self.output(name)
    
    def run_one_step(self, u0:TensorLike, p0:TensorLike, output:bool=False):
        """单步求解"""
        pde = self.equation.pde
        uspace = self.method.uspace
        pspace = self.method.pspace
        threshold = self.method.threshold
        
        BCu = DirichletBC(space=uspace, 
            gd=pde.velocity, 
            threshold=pde.is_u_boundary, 
            method='interp')

        BCp = DirichletBC(space=pspace, 
            gd=pde.pressure, 
            threshold=pde.is_p_boundary, 
            method='interp')
        
        u1 = u0.space.function()
        us = u0.space.function()
        p1 = p0.space.function()
       

        A0, b0 = self.method.predict_velocity(u0, p0, BC=BCu, threshold=threshold, return_form=False)
        us[:] = self.solve(A0, b0)

        A1, b1 = self.method.pressure(u0, p0, BC=BCp, return_form=False)
        p1[:] = self.solve(A1, b1)

        A2, b2 = self.method.correct_velocity(us, p0, p1, return_form=False)
        u1[:] = self.solve(A2, b2)
        return u1, p1

    def output(self, name='result.vtu'):
        """输出当前的结果"""
        from ....functionspace import TensorFunctionSpace
        equation = self.equation
        mesh = self.method.mesh
        output_params = self.params.output_params
        path = output_params["path"]
        coef = output_params["coef"]
        if not name.endswith('.vtu'):
            raise ValueError("输出文件必须以 .vtu 结尾")
        fname = path + name


        for var_name, value in equation._variables.items():
            value_space = value.space
            if isinstance(value_space, TensorFunctionSpace):
                mesh.nodedata[var_name] = value.reshape(value.shape).T
            else:
                mesh.nodedata[var_name] = value

        if coef:
            for var_name, value in equation._coefs.items():
                if callable(value):
                    value_space = value.space
                    if isinstance(value_space, TensorFunctionSpace):
                        mesh.nodedata[var_name] = value.reshape(value.shape).T
                    else:
                        mesh.nodedata[var_name] = value
                else:
                    print(f"{var_name}: {value}")

        mesh.to_vtk(fname=fname)



class IPCSSimulationParameters(SimulationParameters):
    def __init__(self):
        # 核心参数结构
        self._params = {
            "time": {
                "T0": 0.0,                  # 初始时间
                "T1": 1.0,                  # 终止时间
                "dt": 0.01,                 # 时间步长
                "nt": 100                  # 时间层
            },
            "output": {
                "path": "./",            # 路径
                "coef": False,            # 是否输出系数
                "onoff": True           # 输出开关
            },
            "solver": {
                "type": 'direct',  # 求解器类型
                "params": 'mumps',       # 接口类型
            },
        }

    @property
    def timeline(self) :
        return self._params["output"]

    @property
    def output_params(self) :
        """获取输出参数（只读视图）"""
        return self._params["output"]

    @property
    def solver_type(self):
        """获取求解器类型（带类型提示）"""
        return self._params["solver"]

    def set_output(self, *, fname: str = None, coef: float = None, onoff: bool = None):
        """安全设置输出参数"""
        if fname is not None:
            self._params["output"]["fname"] = fname
            
        if coef is not None:
            self._params["output"]["coef"] = float(coef)
            
        if onoff is not None:
            self._params["output"]["onoff"] = bool(onoff)
    
    def set_solver(self, solver_type, api: str = None):
        """设置求解器参数，支持自定义求解器"""
        VALID_SOLVER_TYPES = {"direct", "iterative"}
        if isinstance(solver_type, str):
            if solver_type not in VALID_SOLVER_TYPES:
                raise ValueError(f"求解器类型必须是 {VALID_SOLVER_TYPES} 或非字符串的自定义对象，得到 {solver_type}")
            self._params["solver"]["type"] = solver_type
            if api:
                self._params["solver"]["params"] = api.lower()
        else:
            if not callable(solver_type):
                raise TypeError(f"自定义求解器必须是可调用对象，得到 {type(solver_type)}")
            sig = inspect.signature(solver_type)
            params = list(sig.parameters.keys())
            if len(params) < 2 or params[:2] != ['A', 'b']:
                raise ValueError(f"自定义求解器必须接受 'A' 和 'b' 作为前两个参数，得到 {params}")
            self._params["solver"]["type"] = "custom"
            self._params["solver"]["params"] = solver_type
    
    def set_time(self, *, T0: float = None, T1: float = None, dt: float = None, nt: int = None):
        """安全设置时间参数"""
        time = self._params["time"]

        if T0 is not None:
            if not isinstance(T0, (int, float)):
                raise TypeError(f"T0 must be a number, got {type(T0)}")
            time["T0"] = float(T0)

        if T1 is not None:
            if not isinstance(T1, (int, float)):
                raise TypeError(f"T1 must be a number, got {type(T1)}")
            time["T1"] = float(T1)

        if dt is not None:
            if not isinstance(dt, (int, float)) or dt <= 0:
                raise ValueError(f"dt must be a positive number, got {dt}")
            time["dt"] = float(dt)

        if nt is not None:
            if not isinstance(nt, int) or nt <= 0:
                raise ValueError(f"nt must be a positive integer, got {nt}")
            time["nt"] = nt
        
        if time["nt"] is None and time["dt"] is not None:
            time["nt"] = int((time["T1"] - time["T0"]) / time["dt"]) + 1


        # 验证时间参数的合理性
        if T0 is not None or T1 is not None:
            if time["T0"] >= time["T1"]:
                raise ValueError(f"T0 ({time['T0']}) must be less than T1 ({time['T1']})")




