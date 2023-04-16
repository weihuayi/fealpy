import numpy as np
import matplotlib.pyplot as plt

class NonlinearSolver:
    def __init__(self, tol, max_iter):
        self.tol = tol
        self.max_iter = max_iter

    def newton_raphson(self, u, f, calculate_P, calculate_Kt):
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        R = f - P
        conv = np.sum(R**2)/(1+np.sum(f**2))

        if len(u) == 1:
            print('iter   u1          conv      c')
            print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
        else:
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

            if len(u) == 1:
                print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
            else:
                print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        return u

    def newton_raphson_scalar_with_plot(self, u0, tol, max_iter, calculate_P, calculate_Kt):
        xdata = [0] * 40
        ydata = [0] * 40
        iter = 0
        u = u0
        uold = u

        P = calculate_P(u)
        R = -P
        conv = R**2
        xdata[0] = u
        ydata[0] = P

        while conv > tol and iter < max_iter:
            Kt = calculate_Kt(u)
            delu = R / Kt
            u = uold + delu
            P = calculate_P(u)
            R = -P
            conv = R**2
            uold = u
            iter += 1
            xdata.insert(2 * iter, u)
            ydata.insert(2 * iter, 0)
            xdata.insert(2 * iter + 1, u)
            ydata.insert(2 * iter + 1, P)

        xdata = np.array(xdata[:40])
        ydata = np.array(ydata[:40])

        plt.plot(xdata, ydata, 'o-')
        x = np.arange(-1, 1, 0.1)
        y = x + np.arctan(5 * x)
        plt.plot(x, y)
        plt.show()

        return u


    def modified_newton_raphson(self, u, f, calculate_P, calculate_Kt):
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        R = f - P
        conv = np.sum(R**2)/(1+np.sum(f**2))

        if len(u) == 1:
            print('iter   u1          conv      c')
            print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
        else:
            print('iter   u1      u2          conv      c')
            print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        Kt = calculate_Kt(u)
        while conv > self.tol and iter < self.max_iter:
            delu = np.linalg.solve(Kt, R)
            u = uold + delu
            P = calculate_P(u)
            R = f - P
            conv = np.sum(R**2)/(1+np.sum(f**2))
            c = abs(0.9-u[1])/abs(0.9-uold[1])**2 if len(u) > 1 else 0
            uold = u
            iter += 1

            if len(u) == 1:
                print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
            else:
                print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        return u


    def incremental_secant(self, u, f, calculate_P, calculate_Kt):
        iter = 0
        c = 0
        uold = u
        P = calculate_P(u)
        Pold = P
        R = f - P
        if f == 0:
            conv = np.sum(R**2)
        else:
            conv = np.sum(R**2)/(1+np.sum(f**2))

        if len(u) == 1:
            print('iter   u1          conv      c')
            print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
        else:
            print('iter   u1      u2          conv      c')
            print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        Ks = calculate_Kt(u)
        while conv > self.tol and iter < self.max_iter:
            delu = R / Ks # delu 2D-ndarray 
            u = uold + delu.reshape(-1)
            P = calculate_P(u)
            R = f - P
            if f == 0:
                conv = np.sum(R**2)
            else:
                conv = np.sum(R**2)/(1+np.sum(f**2))
            c = abs(u[0]) / abs(uold[0])**2 
            Ks = (P-Pold) / (u-uold) # 割线矩阵 
            uold = u
            Pold = P
            iter += 1

            if len(u) == 1:
                print(f'{iter:3d} {u[0]:7.5f} {conv:12.3e} {c:7.5f}')
            else:
                print(f'{iter:3d} {u[0]:7.5f} {u[1]:7.5f} {conv:12.3e} {c:7.5f}')

        return u
    # ... 其他方法 ...
