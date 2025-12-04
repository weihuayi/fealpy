from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['RasingBubble2D']

class RasingBubble2D(CNodeType):
    r"""2D Rising Bubble Problem.
    Inputs:
        domain (list): Computational domain, [x0,x1,y0,y1].
        d (float): Diameter of the area.
        area (float): Area of the bubble.
        rho0 (float): Density of fluid 0.
        rho1 (float): Density of fluid 1.
        mu0 (float): Viscosity of fluid 0.
        mu1 (float): Viscosity of fluid 1.
        epsilon (float): Thickness of the interface.
    Outputs:
        velocity_dirichlet_bc (function): Velocity Dirichlet boundary condition.
        init_phase (function): Initial phase function.
        init_velocity (function): Initial velocity.
        init_pressure (function): Initial pressure.
        phase_force (function): Phase function source term.
        velocity_force (function): Velocity source term.
        rho0 (float): Dimensionless density of fluid 0.
        rho1 (float): Dimensionless density of fluid 1.
        mu0 (float): Dimensionless viscosity of fluid 0.
        mu1 (float): Dimensionless viscosity of fluid 1.
        epsilon (float): Dimensionless thickness of the interface.
        box (list): Dimensionless computational domain.
        area (float): Dimensionless area of the bubble.
    """
    TITLE: str = "二维两相流气泡上升模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点构建二维两相流气泡上升问题的数学模型, 定义速度的Dirichlet边界条件、初始相场函数、初始速度场、初始压力场、
                相场函数源项以及速度源项等，用于后续两相流有限元求解。"""
    INPUT_SLOTS = [
        PortConf("domain", DataType.NONE, title="计算域"),
        PortConf("d", DataType.FLOAT, title="区域直径"),
        PortConf("area", DataType.FLOAT, title="区域面积"),
        PortConf("rho0", DataType.FLOAT, title="第一液相密度"),
        PortConf("rho1", DataType.FLOAT, title="第二液相密度"),
        PortConf("mu0", DataType.FLOAT, title="第一液相粘度"),
        PortConf("mu1", DataType.FLOAT, title="第二液相粘度"),
        PortConf("epsilon", DataType.FLOAT, title="界面厚度"),
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity_dirichlet_bc", DataType.FUNCTION, title="速度边界条件"),
        PortConf("init_phase", DataType.FUNCTION, title="初始相场函数"),
        PortConf("init_velocity", DataType.FUNCTION, title="初始速度"),
        PortConf("init_pressure", DataType.FUNCTION, title="初始压力"),
        PortConf("phase_force", DataType.FUNCTION, title="相场源项"),
        PortConf("velocity_force", DataType.FUNCTION, title="速度源项"),
        PortConf("rho0", DataType.FLOAT, title="无量纲第一液相密度"),
        PortConf("rho1", DataType.FLOAT, title="无量纲第二液相密度"),
        PortConf("mu0", DataType.FLOAT, title="无量纲第一液相粘度"),
        PortConf("mu1", DataType.FLOAT, title="无量纲第二液相粘度"),
        PortConf("epsilon", DataType.FLOAT, title="无量纲界面厚度"),
        PortConf("box", DataType.NONE, title="无量纲计算域"),
        PortConf("area", DataType.FLOAT, title="无量纲区域面积"),
    ]
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.decorator import cartesian
        
        class RasingBubble2DModel():
            def __init__(self,domain, d, area, rho0, rho1, 
                      mu0, mu1, epsilon):
                """
                Parameters
                    domain : list
                        Computational domain, [x0,x1,y0,y1].
                    d : float
                        Diameter of the area.
                    area : float
                    rho1 : float
                        Density of fluid 1.
                    rho2 : float
                        Density of fluid 2.
                    mu1 : float
                        Viscosity of fluid 1.
                    mu2 : float
                        Viscosity of fluid 2.
                    epsilon : float
                        Thickness of the interface.
                """
                self.box = domain
                self.d = d
                self.area = area
                self.rho0 = rho0
                self.rho1 = rho1
                self.mu0 = mu0
                self.mu1 = mu1
                self.epsilon = epsilon
                self.g = 9.8
                
                self.dimensionless()
                
            def dimensionless(self):
                ref_length = self.d
                ref_velocity = (self.g*self.d)**0.5
                ref_rho = min(self.rho0,self.rho1)
                ref_mu = ref_rho*ref_length*ref_velocity
                
                self.area /= ref_length**2
                self.box = [x / ref_length for x in self.box]
                self.epsilon /= ref_length

                self.rho0 /= ref_rho
                self.rho1 /= ref_rho
                self.mu0 /= ref_mu
                self.mu1 /= ref_mu
                
                self.d /= ref_length
                self.g = 1.0
            
            @cartesian
            def init_phase(self , p):
                """
                Initial phase function.
                """
                x = p[...,0]
                y = p[...,1]
                r = bm.sqrt(x**2 + y**2)
                val = -bm.tanh((r - 0.5*self.d)/ (self.epsilon))
                return val
            
            @cartesian
            def phase_force(self, p, t):
                """
                Phase function source term.
                """
                x = p[...,0]
                return bm.zeros_like(x, dtype=bm.float64)
            
            @cartesian
            def init_velocity(self, p):
                """
                Initial velocity.
                """
                val = bm.zeros(p.shape, dtype=bm.float64)
                return val
            
            @cartesian
            def velocity_force(self, p, t):
                """
                Velocity source term.
                """
                val = bm.zeros(p.shape, dtype=bm.float64)
                val[...,1] = -self.g
                return val
            
            @cartesian
            def velocity_dirichlet_bc(self, p, t):
                """
                Velocity Dirichlet boundary condition.
                """
                val = bm.zeros(p.shape, dtype=bm.float64)
                return val
            
            @cartesian
            def init_pressure(self, p):
                """
                Initial pressure.
                """
                val = bm.zeros(p.shape[0], dtype=bm.float64)
                return val
        
        model = RasingBubble2DModel(**options)
        return tuple(
            getattr(model, name)
            for name in ["velocity_dirichlet_bc", "init_phase", "init_velocity", "init_pressure",
                         "phase_force", "velocity_force" , 
                         "rho0", "rho1", "mu0", "mu1", "epsilon", "box","area"])
        