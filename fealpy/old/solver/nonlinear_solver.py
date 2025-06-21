import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Tuple 

class NonlinearSolver:
    def __init__(self, tol: np.float_, max_iter: np.int_):
        self.tol = tol
        self.max_iter = max_iter

    def newton_raphson_bivariate(self, u: np.ndarray, f: np.ndarray, 
                       calculate_P: Callable[[np.ndarray], np.ndarray], 
                       calculate_Kt: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        使用 Newton-Raphson 方法求解非线性方程组

        @param u: 初始向量值
        @param f: 非线性方程组右侧的常数向量
        @param calculate_P: 计算非线性方程组的函数
        @param calculate_Kt: 计算切线刚度矩阵的函数
        @return: 非线性方程组的解
        @TODO Delete
        """
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        R = f - P
        conv = np.sum(R**2)/(1+np.sum(f**2))

        print('iter   u1      u2          conv      c')
        print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        while conv > self.tol and iter < self.max_iter:
            Kt = calculate_Kt(u)
            delu = np.linalg.solve(Kt, R)
            u = uold + delu
            P = calculate_P(u)
            R = f - P
            conv = np.sum(R**2)/(1+np.sum(f**2))
            c = abs(0.9-u[1])/abs(0.9-uold[1])**2 if len(u) > 1 else 0
            uold = u
            iter += 1

            print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        return u


    def newton_raphson(self, u: np.ndarray, f: np.ndarray, 
                   calculate_P: Callable[[np.ndarray], np.ndarray], 
                   calculate_Kt: Callable[[np.ndarray], np.ndarray], 
                   u_exact: np.ndarray) -> np.ndarray:
        """
        使用 Newton-Raphson 方法求解非线性方程组

        @param u: 初始向量值
        @param f: 非线性方程组右侧的常数向量
        @param calculate_P: 计算非线性方程组的函数
        @param calculate_Kt: 计算切线刚度矩阵的函数
        @param u_exact: 准确的解向量
        @return: 非线性方程组的解
        """
        iter = 0
        uold = u
        P = calculate_P(u)
        R = f - P
        conv = np.sum(R**2)/(1+np.sum(f**2))
        c = 0

        def print_info():
            print(f'{iter:3d}', end='')
            for ui in u:
                print(f' {ui:7.5f}', end='')
            print(f' {conv:12.3e} {c:7.5f}')

        print('iter', end='')
        for i in range(len(u)):
            print(f'   u{i+1}   ', end='')
        print('      conv      c')
        print_info()

        while conv > self.tol and iter < self.max_iter:
            Kt = calculate_Kt(u)
            delu = np.linalg.solve(Kt, R)
            u = uold + delu
            P = calculate_P(u)
            R = f - P
            conv = np.sum(R**2)/(1+np.sum(f**2))
            c = np.abs(u_exact[1] - u[1])/np.abs(u_exact[1] - uold[1])**2 if iter > 0 else 0
            uold = u
            iter += 1

            print_info()

        return u


    def newton_raphson_unvariate(self, u0: float, calculate_P: Callable[[float], float], 
                                    calculate_Kt: Callable[[float], float]) -> float:
        """
        使用 Newton-Raphson 方法求解单变量非线性方程并绘制收敛过程。

        @param u0: 初始近似解
        @param tol: 收敛容差
        @param max_iter: 最大迭代次数
        @param calculate_P: 计算非线性方程的函数
        @param calculate_Kt: 计算切线斜率的函数
        @return: 非线性方程的解
        @TODO: Delete
        """        
        xdata = [0] * 40 # 初始化一个长度为 40 的列表，用于存储 x 坐标值。
        ydata = [0] * 40 # 初始化一个长度为 40 的列表，用于存储 y 坐标值。
        iter = 0
        u = u0
        uold = u
        P = calculate_P(u)
        R = -P
        conv = R**2
        xdata[0] = u
        ydata[0] = P

        while conv > self.tol and iter < self.max_iter:
            Kt = calculate_Kt(u)
            delu = R / Kt
            u = uold + delu
            P = calculate_P(u)
            R = -P
            conv = R**2
            uold = u
            iter += 1
            
            # 将每次迭代的结果添加到 xdata 和 ydata 中：
            xdata.insert(2 * iter, u) # 在 xdata 的 2*iter 位置插入当前迭代解 u
            ydata.insert(2 * iter, 0) # 在 xdata 的 2*iter 位置插入当前迭代解 u
            xdata.insert(2 * iter + 1, u) # 在 xdata 的 2*iter+1 位置插入当前迭代解 u
            ydata.insert(2 * iter + 1, P)# 在 xdata 的 2*iter+1 位置插入当前迭代非线性方程值 P

        xdata = np.array(xdata[:40])
        ydata = np.array(ydata[:40])

        plt.plot(xdata, ydata, 'o-')
        x = np.arange(-1, 1, 0.1)
        y = x + np.arctan(5 * x)
        plt.plot(x, y)
        plt.show()

        return u


    def modified_newton_raphson(self, u: np.ndarray, f: np.ndarray, 
                       calculate_P: Callable[[np.ndarray], np.ndarray], 
                       calculate_Kt: Callable[[np.ndarray], np.ndarray],
                       u_exact: np.ndarray) -> np.ndarray:
        """
        使用修正的 Newton-Raphson 方法求解非线性方程组

        @param u: 初始向量值
        @param f: 非线性方程组右侧的常数向量
        @param calculate_P: 计算非线性方程组的函数
        @param calculate_Kt: 计算切线刚度矩阵的函数
        @param u_exact: 准确的解向量
        @return: 非线性方程组的解
        """
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        R = f - P
        conv = np.sum(R**2)/(1+np.sum(f**2))

        def print_info():
            print(f'{iter:3d}', end='')
            for ui in u:
                print(f' {ui:7.5f}', end='')
            print(f' {conv:12.3e} {c:7.5f}')

        print('iter', end='')
        for i in range(len(u)):
            print(f'   u{i+1}   ', end='')
        print('      conv      c')
        print_info()

        Kt = calculate_Kt(u)
        while conv > self.tol and iter < self.max_iter:
            delu = np.linalg.solve(Kt, R)
            u = uold + delu
            P = calculate_P(u)
            R = f - P
            conv = np.sum(R**2)/(1+np.sum(f**2))
            c = abs(u_exact[1]-u[1]) / abs(u_exact[1]-uold[1])**2 if len(u) > 1 else 0
            uold = u
            iter += 1

            print_info()

        return u


    def incremental_secant_unvariate(self, u0: float, calculate_P: Callable[[float], float], 
                                      calculate_Kt: Callable[[float], float]) -> float:
        """
        使用增量割线法求解单变量非线性方程并绘制收敛过程。

        @param u0: 初始近似解
        @param calculate_P: 计算非线性方程的函数
        @param calculate_Kt: 计算割线斜率的函数
        @return: 非线性方程的解
        """        
        xdata = [0] * 40 # 初始化一个长度为 40 的列表，用于存储 x 坐标值。
        ydata = [0] * 40 # 初始化一个长度为 40 的列表，用于存储 y 坐标值。
        iter = 0
        c = 0
        u = u0
        uold = u

        P = calculate_P(u)
        Pold = P
        R = -P
        conv = R**2
        xdata[0] = u
        ydata[0] = P
        
        print("\n iter   u          conv      c")
        print(f"\n {iter:3d} {u:7.5f} {conv:12.3e} {c:7.5f}")
        
        Ks = calculate_Kt(u)
        while conv > self.tol and iter < self.max_iter:
            delu = R / Ks # delu 是一个二维数组 
            u = uold + delu
            P = calculate_P(u)
            R = - P
            conv = R**2
            c = abs(u) / abs(uold)**2
            Ks = (P - Pold) / (u - uold) # 割线矩阵 
            uold = u
            Pold = P
            iter += 1
            print(f"\n {iter:3d} {u:7.5f} {conv:12.3e} {c:7.5f}")
            
            # 将每次迭代的结果添加到 xdata 和 ydata 中：
            xdata.insert(2 * iter, u) # 在 xdata 的 2*iter 位置插入当前迭代解 u
            ydata.insert(2 * iter, 0) # 在 xdata 的 2*iter 位置插入当前迭代解 u
            xdata.insert(2 * iter + 1, u) # 在 xdata 的 2*iter+1 位置插入当前迭代解 u
            ydata.insert(2 * iter + 1, P)# 在 xdata 的 2*iter+1 位置插入当前迭代非线性方程值 P

        xdata = np.array(xdata[:40])
        ydata = np.array(ydata[:40])

        plt.plot(xdata, ydata, 'o-')
        x = np.arange(-2, 2, 0.1)
        y = x + np.arctan(5 * x)
        plt.plot(x, y)
        plt.show()

        return u


    def Broyden(self, u: np.ndarray, f: np.ndarray, 
                       calculate_P: Callable[[np.ndarray], np.ndarray], 
                       calculate_Kt: Callable[[np.ndarray], np.ndarray],
                       u_exact: np.ndarray) -> np.ndarray:
        """
        使用 Broyden 方法求解非线性方程组

        @param u: 初始向量值
        @param f: 非线性方程组右侧的常数向量
        @param calculate_P: 计算非线性方程组的函数
        @param calculate_Kt: 计算切线刚度矩阵的函数
        @param u_exact: 准确的解向量
        @return: 非线性方程组的解
        """
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        R = P - f
        Rold = R
        conv = np.sum(R ** 2) / (1 + np.sum(f**2))

        def print_info():
            print(f'{iter:3d}', end='')
            for ui in u:
                print(f' {ui:7.5f}', end='')
            print(f' {conv:12.3e} {c:7.5f}')

        print('iter', end='')
        for i in range(len(u)):
            print(f'   u{i+1}   ', end='')
        print('      conv      c')
        print_info()

        Ks = calculate_Kt(u)
        while conv > self.tol and iter < self.max_iter:
            delu = -np.linalg.solve(Ks, R)
            u = uold + delu
            P = calculate_P(u)
            R = P - f
            conv = np.sum(R ** 2) / (1 + np.sum(f ** 2))
            c = abs(u_exact[1]-u[1]) / abs(u_exact[1]-uold[1])**2 if len(u) > 1 else 0
            delR = R - Rold
            Ks = Ks + (delR - Ks @ delu)[:, None] * delu[None, :] / np.linalg.norm(delu) ** 2
            uold = u
            Rold = R
            iter += 1

            print_info()

        return u

    
    def Incremental_Force(self, u_init: np.ndarray, f: np.ndarray, force_increment: np.int_,
                        calculate_P: Callable[[np.ndarray], np.ndarray], 
                        calculate_Kt: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        使用增量力法求解非线性方程组

        @param u_init: 初始向量值
        @param f: 非线性方程组右侧的常数向量
        @param force_increment: 载荷增量数量
        @param calculate_P: 计算非线性方程组的函数
        @param calculate_Kt: 计算切线刚度矩阵的函数
        @return: 非线性方程组的解
        """
        u = u_init
        uold = u
        f_increment = f / force_increment
        print("步骤   u1        u2        F")

        # 存储数据的列表
        u1_list = []
        u2_list = []
        F_list = []

        for i in range(1, force_increment+1):
            f_current = f_increment * i
            P = calculate_P(u)
            R = f_current - P
            conv = np.sum(R**2) / (1+np.sum(f_current**2))
            
            iter = 0
            while conv > self.tol and iter < self.max_iter:
                Kt = calculate_Kt(u)
                delu = np.linalg.solve(Kt, R)
                u = uold + delu
                P = calculate_P(u)
                R = f_current - P
                conv = np.sum(R**2)/(1+np.sum(f_current**2))
                uold = u
                iter += 1

            print("{:3d} {:7.5f} {:7.5f} {:7.3f}".format(i, u[0], u[1], f_current[1]))

            # 将数据添加到列表中
            u1_list.append(u[0])
            u2_list.append(u[1])
            F_list.append(f_current[1])

        # 绘制 u1、u2 与 F 的关系图
        plt.figure()
        plt.plot(F_list, u1_list, marker='o', label='u1')
        plt.plot(F_list, u2_list, marker='s', label='u2')
        plt.xlabel('Force')
        plt.ylabel('Displacement')
        plt.legend()
        plt.grid(True)
        plt.show()
        return u


    def Incremental_displacement(self, u1_init: np.float_, u2_init: np.float_,
                    displacement_increment: np.int_,
                    calculate_P: Callable[[np.float_, np.float_], np.float_],
                    calculate_Kt: Callable[[np.float_, np.float_], np.float_],
                    calculate_F: Callable[[np.float_, np.float_], np.float_]) -> Tuple[np.float_, np.float_]:
        """
        使用增量位移法求解非线性方程组

        @param u1_init: u1 的初始值
        @param u2_init: u2 的初始值
        @param displacement_increment: 位移增量数量
        @param calculate_P: 计算非线性方程组 P 的函数
        @param calculate_Kt: 计算切线刚度矩阵 Kt 的函数
        @param calculate_F: 计算作用力 F 的函数
        @return: 非线性方程组的解 (u1, u2)
        """
        u1 = u1_init
        u1old = u1
        print("步骤   u1        u2        F")

        # 存储数据的列表
        u1_list = []
        u2_list = []
        F_list = []
        
        # 位移循环增量
        for i in range(1, displacement_increment):
            u2 = u2_init * i
            P = calculate_P(u1, u2)
            R = - P
            conv = R ** 2
            
            # 收敛循环
            iter = 0
            while conv > self.tol and iter < self.max_iter:
                Kt = calculate_Kt(u1, u2)
                delu1 = R / Kt
                u1 = u1old + delu1
                P = calculate_P(u1, u2)
                R = -P
                conv = R**2
                u1old = u1
                iter += 1

            F = calculate_F(u1, u2)
            print("{:3d} {:7.5f} {:7.5f} {:7.3f}".format(i, u1, u2, F))

            # 将数据添加到列表中
            u1_list.append(u1)
            u2_list.append(u2)
            F_list.append(F)

        # 绘制 u1、u2 与 F 的关系图
        plt.figure()
        plt.plot(F_list, u1_list, marker='o', label='u1')
        plt.plot(F_list, u2_list, marker='s', label='u2')
        plt.xlabel('Force')
        plt.ylabel('Displacement')
        plt.legend()
        plt.grid(True)
        plt.show()

        return u1, u2
    # ... 其他方法 ...
