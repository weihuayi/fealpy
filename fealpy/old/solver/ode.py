import numpy as np


class ODESolver():
    def run(self, x, t, dt, tf):
        """
        @brief 从时刻 t 到 tf 时间积分

        @param[in, out]
        """
        while t < tf:
            self.step(x, t, dt)


class ForwardEulerSovler(ODESolver):

    def __init__(self, f):
        self.f = f # 时间依赖的算子 f(x, t): R^n --> R^m
        self.dxdt = np.zeros(f.m, dtype=f.dtype)

    def step(self, x, t, dt):
        """
        """
        self.f.set_time(t)
        self.dxdt[:] = self.f@x
        x += dt*self.dxdt
        t += dt

class ImplicitMidpointSolver(ODESolver):

    def __init__(self, f):
        self.f = f
        self.k = np.zeros(f.shape[0], dtype=f.dtype)

    def step(self, x, t, dt):
        """
        """
        f = self.f
        k = self.k

        f.set_time(t + dt/2)
        f.implicit_solve(dt/2, x, k)
        x += dt*k
        t += dt

class BackwardEulerSolver(ODESolver):
    """
    @brief 向后欧拉方法， L-stable
    """
    def __init__(self, f):
        self.f = f
        self.k = np.zeros(f.shape[0], dtype=f.dtype)

    def step(self, x, t, dt):
        """
        """
        f = self.f
        k = self.k
        f.set_time(t + dt)
        f.implicit_solve(dt, x, k) 
        x += dt*k



class RK2Solver(ODESolver):
    """
    @brief 显式 2 阶 Runge-Kutta 方法

    a = 1/2 中点方法
    a = 1   Heun 方法
    a = 2/3 默认方法，最小截断误差
    """
    def __init__(self, f, a=2.0/3.0):
        self.f = f # 时间依赖的算子 f(x, t): R^n --> R^m
        self.a = a
        self.dxdt = np.zeros(f.shape[0], dtype=f.dtype)
        self.x1 = np.zeros(f.shape[0], dtype=f.dtype)

    def step(self, x, t, dt):
        """
        0 |
        a |  a
        --+--------
          | 1-b  b       b = 1/(2a)
        """
        f = self.f
        a = self.a
        dxdt = self.dxdt
        x1 = self.x1

        b = 0.5/a
        f.set_time(t)
        dxdt[:] = f@x
        x1[:] = x + (1.0 - b)*dt*dxdt
        x += a*dt*dxdt 

        f.set_time(t + a*dt)
        dxdt[:] = f@x
        x[:] = x1 + b*dt*dxdt 
        t += dt


class RK3Solver(ODESolver):
    """
    @brief 保持强稳定性的三阶Runge Kutta方法
    """

    def __init__(self, f):
        self.f = f
        self.y = np.zeros(f.shape[0], dtype=f.dtype) 
        self.k = np.zeros(f.shape[0], dtype=f.dtype)

    def step(self, x, t, dt):
        """
        @brief 
        """
        f = self.f
        y = self.y
        k = self.k
        
        # x0 = x, t0 = t, k0 = dt*f(t0, x0) 
        f.set_time(t)
        f.mv(x, k)

        # x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
        y[:] = x + dt*k
        f.set_time(t + dt)
        f.mv(y, k)

        # x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
        y += dt*k
        y[:] = 0.75*x + 0.25*y
        f.set_time(t + dt/2.0)
        f.mv(y, k)


        # x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
        y += dt*k
        x[:] = 1./3*x + 2./3*y 

        t += dt


class RK4Solver(ODESolver):
    """
    @brief 经典四阶 Runge-Kutta 方法
    """

    def __init__(self, f):
        self.f = f
        self.y = np.zeros(f.shape[0], dtype=f.dtype) 
        self.k = np.zeros(f.shape[0], dtype=f.dtype)
        self.z = np.zeros(f.shape[0], dtype=f.dtype) 

    def step(self, x, t, dt):
        """
        0  |
       1/2 | 1/2
       1/2 |  0   1/2
        1  |  0    0    1
      -----+-------------------
           | 1/6  1/3  1/3  1/6
        """
        f = self.f
        y = self.y
        k = self.k
        z = self.z

        f.set_time(t)
        f.mv(x, k)# k1
        y[:] = x + dt/2*k
        z[:] = x + dt/6*k

        f.set_time(t + dt/2)
        f.mv(y, k) # k2
        y[:] = x + dt/2*k
        z += dt/3*k

        k[:] = f.mv(y, k) # k3
        y[:] = x + dt*k
        z += dt/3*k

        f.set_time(t + dt)
        f.mv(y, k) # k4
        x[:] = z + dt/6*k
        t += dt


