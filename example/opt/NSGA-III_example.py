from fealpy.opt import NondominatedSortingGeneticAlgIII
from fealpy.opt.model import OPTModelManager
from fealpy.opt.opt_function import generate_reference_points_double_layer, initialize

M = 3 
dim = 7
options = {
    'n_obj': M,
    'n_var': dim
}

# 1. 实例化模型
manager = OPTModelManager('multi')
text = manager.get_example(1, **options)

# 2. 上下界
lb, ub = text.get_bounds()

# 3. 生成参考点
zr = generate_reference_points_double_layer(M=M, H1=3, H2=2)

# 4. 初始化种群
NP = zr.shape[0]
x = initialize(zr.shape[0], dim, ub, lb)

# 5. 包装目标函数
fobj = lambda x: text.evaluate(x)

# 6. 初始化算法
test = NondominatedSortingGeneticAlgIII(
    M=M, 
    dim=dim, 
    lb=lb, 
    ub=ub, 
    zr=zr, 
    x=x, 
    fobj=fobj
)

# 7. 优化开始
pop = test.run()

# 8. 输出指标
print(test.cal_IGD())
print(test.cal_spacing())