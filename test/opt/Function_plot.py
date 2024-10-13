# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Function import Function
from fealpy.backend import backend_manager as bm

# bm.set_backend('pytorch')
#测试函数绘图
class Function_plot:

    def __init__(self, F):
        self.F = F
    
    def Functions_plot(self):
        plt.clf()  # 清除当前图形
        plt.close()  # 关闭当前图形窗口

        f_obj = Function(self.F)
        fobj, lb, ub, dim = f_obj.Functions()

        if self.F == 'F1':
            x = bm.arange(-5.12, 5.12, 0.1)
            y = bm.arange(-5.12, 5.12, 0.1)
        elif self.F == 'F2':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F3':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F4':
            x = bm.arange(-1.28, 1.28, 0.1)
            y = bm.arange(-1.28, 1.28, 0.1)
        elif self.F == 'F5':
            x = bm.arange(-4.5, 4.5, 0.1)
            y = bm.arange(-4.5, 4.5, 0.1)
        elif self.F == 'F6':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F7':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F8':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F9':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F10':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F11':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F12':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F13':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F14':
            x = bm.arange(-10, 10, 0.1)
            y = bm.arange(-10, 10, 0.1)
        elif self.F == 'F15':
            x = bm.arange(-bm.pi, bm.pi, 0.1)
            y = bm.arange(-bm.pi, bm.pi, 0.1)
        elif self.F == 'F16':
            x = bm.arange(-bm.pi, bm.pi, 0.1)
            y = bm.arange(-bm.pi, bm.pi, 0.1)
        elif self.F == 'F17':
            x = bm.arange(-bm.pi, bm.pi, 0.1)
            y = bm.arange(-bm.pi, bm.pi, 0.1)
        elif self.F == 'F18':
            x = bm.arange(-5.12, 5.12, 0.1)
            y = bm.arange(-5.12, 5.12, 0.1)
        elif self.F == 'F19':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F20':
            x = bm.arange(-5, 5, 0.01)
            y = bm.arange(-5, 5, 0.01)
        elif self.F == 'F21':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F22':
            x = bm.arange(-100, 100, 1)
            y = bm.arange(-100, 100, 1)
        elif self.F == 'F23':
            x = bm.arange(-10, 10, 1)
            y = bm.arange(-10, 10, 1)
        elif self.F == 'F24':
            x = bm.arange(-30, 30, 0.1)
            y = bm.arange(-30, 30, 0.1)
        elif self.F == 'F25':
            x = bm.arange(-50, 50, 0.1)
            y = bm.arange(-50, 50, 0.1)
        elif self.F == 'F26':
            x = bm.arange(-32, 32, 0.1)
            y = bm.arange(-32, 32, 0.1)
        else:
            raise ValueError("Function not implemented")

        #创建二维网格以及矩阵
        X, Y = np.meshgrid(x, y)
        Z = bm.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                if self.F != 'F8' and self.F != 'F23':
                    Z[i,j] = fobj(bm.array([x[i], y[j]]))
                elif self.F == 'F8':
                    Z[i,j] = fobj(bm.array([x[i], y[j], 0, 0]))
                elif self.F == 'F23':
                    Z[i,j] = fobj(bm.array([x[i], y[j], 0, 0, 0]))
        
        return X, Y, Z


if __name__ == "__main__":     
    print('--------Functions_plot----------------')

    F = 'F1'
    f_plot = Function_plot(F)
    #print(f_plot)

    X, Y, Z = f_plot.Functions_plot()
