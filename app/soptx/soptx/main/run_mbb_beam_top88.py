from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.opt import opt_alg_options

from app.soptx.soptx.cases.mbb_beam_cases import MBBBeamCase
from app.soptx.soptx.utilfunc.fem_solver import FEMSolver
from app.soptx.soptx.opt.oc_optimizer import OCOptimizer
from app.soptx.soptx.cases.termination_criterias import TerminationCriteria
from app.soptx.soptx.cases.top_opt_porblem import ComplianceMinimization
from app.soptx.soptx.utilfunc.sensitivity_calculations import manual_objective_sensitivity
from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.volume_objective import VolumeConstraint
# from app.soptx.soptx.optalg.oc_alg import OCAlg

backend = 'pytorch'
bm.set_backend(backend)

# 初始化 MBB 梁算例
mbb_case = MBBBeamCase(case_name="top88")

# 定义网格
nx, ny = mbb_case.nx, mbb_case.ny
extent = [0, nx, 0, ny]
h = mbb_case.h 
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

# 定义有限元空间
p_C = 1
space_C = LagrangeFESpace(mesh, p=p_C, ctype='C')
tensor_space = TensorFunctionSpace(space_C, shape=(-1, 2))

p_D = 0
space_D = LagrangeFESpace(mesh, p=p_D, ctype='D')        
rho = space_D.function()

rho[:] = mbb_case.rho

# 获取材料属性
material_properties = mbb_case.material_properties
material_properties.rho = rho

# 获取边界条件
boundary_conditions = mbb_case.boundary_conditions

# 获取滤波器属性
filter_properties = mbb_case.filter_properties

# 创建 FEM 求解器
fem_solver = FEMSolver(material_properties=material_properties, 
                       tensor_space=tensor_space,
                       boundary_conditions=boundary_conditions)

# 创建柔顺度目标函数
compliace_objective = ComplianceObjective(mesh=mesh, space=tensor_space,
                                            material_properties=material_properties,
                                            filter_properties=filter_properties,
                                            displacement_solver=fem_solver)

c_value = compliace_objective.fun(rho=material_properties.rho)

dce_value = compliace_objective.jac(rho=material_properties.rho)

# 获取约束条件
constraint_conditions = mbb_case.constraint_conditions
volfrac = constraint_conditions.get_constraints()['volume']['vf']

# 创建体积约束条件
compliance_constraint = VolumeConstraint(mesh=mesh, volfrac=volfrac, 
                                        filter_properties=filter_properties)

cneq = compliance_constraint.fun(rho=material_properties.rho)

gradc = compliance_constraint.jac()



options = opt_alg_options(
    x0=material_properties.rho,
    objective=compliace_objective,
    volume_constraint=compliance_constraint,
    MaxIters=100,
    NormGradTol=1e-6,
    tol_change=0.01  # Set tolerance for design variable change
)

oc_optimizer = OCAlg(options)
oc_optimizer.run()










# 创建优化终止条件
termination_criteria = mbb_case.termination_criterias

# 定义目标函数
objective_function = ComplianceMinimization()

# 创建滤波器参数（如果需要）
filter_properties = mbb_case.filter_properties

constraint_conditions = mbb_case.constraint_conditions

# 创建 OC 优化器
optimizer = OCOptimizer(displacement_solver=fem_solver, 
                        objective_function=objective_function,
                        sensitivity_function=manual_objective_sensitivity,
                        termination_criteria=termination_criteria,
                        constraint_conditions=constraint_conditions,
                        filter_parameters=filter_properties)


# 执行优化
optimized_rho = optimizer.optimize(rho=rho)

# 输出优化结果
print(f"Optimized Density Distribution:\n{optimized_rho}")
