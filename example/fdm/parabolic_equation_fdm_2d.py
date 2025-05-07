from fealpy.fdm.parabolic_operator import ParabolicOperator
from fealpy.fdm import DirichletBC
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh
from fealpy.solver import spsolve
import matplotlib.pyplot as plt
from fealpy.utils import timer
from fealpy.pde.parabolic import ParabolicData
time = timer()
next(time)

# 步进方法
def advance(n, uh_history):
    """
    @brief 时间步进格式为任意方法
    
    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    bc = DirichletBC(mesh,gd = lambda p: pde.dirichlet(p, t+tau))

    source = lambda p: pde.source(p, t + tau)
    f = mesh.interpolate(source) # f.shape = (nx+1,ny+1)
    A, B_tuple = PBO()  # Get the CN assembly matrices

    # 计算右端项 F
    F = B_tuple[-1] * f
    for i in range(len(B_tuple) - 1):
        F += B_tuple[i] @ uh_history[i].flat[:]

    A, F = bc.apply(A, F.flat)  # Apply Dirichlet boundary condition to A and f
    uh0.flat = spsolve(A, F, solver='scipy')

        # 更新 uh_history，存储最新的解
    uh_history.insert(0, uh0.copy())  # 插入最新的解到历史记录
    if len(uh_history) > len(B_tuple) - 1:
        uh_history.pop()  # 保持历史记录的长度与 B_tuple 一致
    solution = lambda p : pde.solution(p, t + tau)

    # 三种误差的计算
    e[0,n],e[1,n],e[2,n] = mesh.error(solution,uh0)
    return uh0, t

# 绘图函数
def plot_solution(solutions, mesh):
    """
    绘制当前时间步的解

    Parameters:
        uh0 array, 当前时间步的解
        t float, 当前时间
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for t, uh in solutions:
        mesh.show_function(ax, uh)
    plt.title("Solutions at Different Time Points")
    plt.xlabel("x")
    plt.ylabel("uh")
    plt.legend()
    plt.show()

def plot_error(errors, time_values,name):
    """
    绘制误差随时间变化的曲线

    Parameters:
        errors array, 误差数组
        time_values array, 时间数组
        name str, 误差类型名称
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_values[1:], errors[1:],
              linestyle='-', color='b', label=name)
    plt.title(f"{name} vs Time")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


# 定义2维域和网格
domain = [0, 1, 0, 1]  # Domain of the mesh
nx = 5
ny = 5  # Number of nodes in x and y directions
hx = (domain[1] - domain[0]) / nx  # Mesh spacing in x-direction
hy = (domain[3] - domain[2]) / ny  # Mesh spacing in y-direction
h = (hx,hy)  # Mesh spacing in each dimension
origin = [domain[0], domain[2]]  # Origin of the mesh
mesh = UniformMesh([0,nx,0,ny], h , origin)  # Create a uniform mesh
# 定义时间范围和时间步长
duration = [0, 1]  # Time duration of the simulation
nt = 3000
tau = (duration[1] - duration[0])/nt
# 定义抛物方程数据
pde = ParabolicData('exp(-20*t)*sin(4*pi*x)*sin(4*pi*y)', 
                      ['x', 'y'], 't', D=domain, T=duration)  # PDE data



time_points = bm.array([ 0.2, 0.4, 0.6, 0.8])  # 需要保存解的时间点

epsilon = 1e-6  # 容差，用于判断浮点数是否接近
maxit = 5 # 加密次数

# 定义误差数组
# e 为每个时间步的误差
# em 为特定时间点的误差,在每个网格细化迭代中保存
e = bm.zeros((3,nt), dtype=bm.float64)  # Error array for storing errors at each time step
em = bm.zeros((3, maxit), dtype=bm.float64)  
error_name = ['max error', 'L2 error', 'l2 error']

uh0 = mesh.interpolate(pde.init_solution)
time.send('init')

# 加密循环
for i in range(maxit):
    solutions = []  # 用于存储解的列表
    time_values = [] # 用于存储时间的列表
    uh0 = mesh.interpolate(pde.init_solution)  
    PBO = ParabolicOperator(mesh, tau, method= 'backward') 
    print(f"Iteration {i+1}")
    uh_history = [uh0.copy() for _ in range(len(PBO()[1]) - 1)]

    # 求解的步进逻辑
    for n in range(nt):
        uh0, t = advance(n, uh_history) # 至此完整求解逻辑已经完成

        # 下面的代码用于记录当前时间的解和误差
        # 记录当前时间和误差
        time_values.append(t)
        # 检查 t 是否接近 time_points 中的任意一个值
        if any(abs(t - tp) < epsilon for tp in time_points):  
            solutions.append((t, uh0.copy()))  # 保存当前时间和解
            # 打印特定时间的三种误差
            print(e[:,n])
    time.send(f'run_iter{i}')
    # 绘制曲线
    plot_solution(solutions, mesh)
    
    for j in range(len(e)):
        errors = e[j]
        plot_error(errors, time_values,error_name[j])
        em[j, i] = e[j, 1200]
    
    mesh.uniform_refine()  # 网格一致加密
next(time)
# 打印误差比
print("em_ratio:\n", em[:, 0:-1]/em[:, 1:])