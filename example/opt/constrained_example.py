from fealpy.opt import QuantumParticleSwarmOpt, initialize, opt_alg_options
from fealpy.opt.model import OPTModelManager
from fealpy.opt.benchmark.constrained_benchmark import constrained_benchmark_data as data

# 1. 初始化 Manager 并获取 benchmark
manager = OPTModelManager('constrained')
text = manager.get_example(1)

# 2. 获取上下界、维度
lb, ub = text.get_bounds() 
dim = text.get_dim()

# 3. 初始化种群
NP = 100 # 种群规模
x0 = initialize(NP, dim, ub, lb, method=None)

# 4. 包装目标函数
fobj = lambda x: text.evaluate(x)

# 5. 构建优化器选项
option = opt_alg_options(x0, fobj, (lb, ub), NP)

# 6. 初始化优化算法
optimizer = QuantumParticleSwarmOpt(option)

# 7. 运行优化
optimizer.run()

# 8. 输出结果
optimizer.print_optimal_result() 
optimizer.plot_curve() 
optimizer.plot_plpt_percen() 
