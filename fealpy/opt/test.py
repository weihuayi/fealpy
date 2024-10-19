import numpy as np
from fealpy.opt.gradient_descent_alg import GradientDescentAlg
from fealpy.opt.optimizer_base import opt_alg_options
from fealpy.opt.line_search_rules import ArmijoLineSearch

# 1. 定义目标函数 (简单的二次函数)
def objective(x):
    """
    Example of a simple quadratic objective function:
    f(x) = (x - 3)^2 + 5
    Gradient: grad(f(x)) = 2*(x - 3)
    """
    f = (x - 3)**2 + 5
    grad = 2 * (x - 3)
    return f, grad

# 2. 设置初始条件和优化参数
x0 = np.array([0.0])  # 起始点，选择任意值，例如 x = 0
StepLength = 1.0      # 步长
MaxIters = 100        # 最大迭代次数
NormGradTol = 1e-6    # 梯度容差
FunValDiff = 1e-6     # 函数值差异容差

# 3. 构造优化器的选项字典                  n
options = opt_alg_options(
    x0=x0,
    objective=objective,
    StepLength=StepLength,
    MaxIters=MaxIters,
    NormGradTol=NormGradTol,
    FunValDiff=FunValDiff,
    LineSearch = ArmijoLineSearch(beta=0.5)
)


# 5. 创建 GradientDescentAlg 实例
GDA = GradientDescentAlg(options)

# 6. 运行优化算法
result_x, result_f, result_g, diff = GDA.run()

# 7. 输出结果
print(f"Optimal x: {result_x}")
print(f"Optimal function value: {result_f}")
print(f"Gradient at optimal point: {result_g}")
print(f"Function value difference: {diff}")
