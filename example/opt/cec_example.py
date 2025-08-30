from fealpy.opt import initialize, opt_alg_options, ParticleSwarmOpt
from fealpy.opt.model import OPTModelManager

dim = 2 # 维度
options = {
    'func_num': 1, # 函数序号
    'dim': dim
}

# 1. 初始化 Manager 并获取 benchmark
manager = OPTModelManager('single')
cec = manager.get_example(3, **options) 

# 2. 获取上下界
lb, ub = cec.get_bounds() 

# 3. 初始化种群
NP = 100 # 种群规模
x0 = initialize(NP, dim, ub, lb, method=None)

# 4. 包装目标函数
fobj = lambda x: cec.evaluate(x)

# 5. 构建优化器选项
option = opt_alg_options(x0, fobj, (lb, ub), NP)

# 6. 初始化优化算法
optimizer = ParticleSwarmOpt(option)

# 7. 运行优化
optimizer.run()

# 8. 输出结果
optimizer.print_optimal_result() 
optimizer.plot_curve() 
optimizer.plot_plpt_percen() 