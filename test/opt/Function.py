import numpy as np
from fealpy.backend import backend_manager as bm

#bm.set_backend('pytorch')
class Function:

    def __init__(self, F):
        self.F = F
    
    #测试函数信息
    def Functions(self): 

        def F1(x):
            return bm.sum((x + 0.5) ** 2)

        def F2(x):
            return bm.sum(x ** 2)
        
        def F3(x):
            dim = x.shape[0]
            return bm.sum(bm.arange(1, dim + 1) * (x ** 2))

        def F4(x):
            dim = x.shape[0]
            return x[0] ** 2 + bm.sum(1e6 * x[1:] ** 2)

        def F5(x):
            return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2

        def F6(x):
            return -bm.cos(x[0]) * bm.cos(x[1]) * bm.exp(-((x[0] - bm.pi) ** 2 + (x[1] - bm.pi) ** 2))

        def F7(x):
            return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

        def F8(x):
            return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2 + (x[2] - 1) ** 2 + 90 * (x[2] ** 2 - x[3]) ** 2 + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1)

        def F9(x):
            dim = x.shape[0]
            return bm.sum(x ** 2) + (bm.sum(0.5 * bm.arange(1, dim + 1) * x)) ** 2 + (bm.sum(0.5 * bm.arange(1, dim + 1) * x)) ** 4

        def F10(x):
            return bm.sum(bm.abs(x)) + bm.prod(bm.abs(x))

        def F11(x):
            dim = x.shape[0]
            o = 0
            for j in range(1, dim + 1):
                oo = 0
                for k in range(1, j + 1):
                    oo += x[k - 1]
                o += oo ** 2
            return o

        def F12(x):
            dim = x.shape[0]
            o = (x[0] - 1) ** 2
            for j in range(2, dim + 1):
                o += j * (2 * x[j - 1] ** 2 - x[j - 1] - 1) ** 2
            return o

        def F13(x):
            return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[0]) - 0.4 * bm.cos(4 * bm.pi * x[1]) + 0.7

        def F14(x):
            return (x[0] + 2 * x[1]) ** 2 + (2 * x[0] + x[1] - 5) ** 2

        def F15(x):
            dim = x.shape[0]
            return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20))  

        def F16(x):
            dim = x.shape[0]
            return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20))

        def F17(x):
            dim = x.shape[0]
            return -bm.sum(bm.sin(x) * (bm.sin(bm.arange(1, dim + 1) * (x ** 2) / bm.pi) ** 20))

        def F18(x):
            n = len(x)
            o = 0
            for i in range(n):
                o += x[i] ** 2 - 10 * bm.cos(2 * bm.pi * x[i]) + 10
            return o

        def F19(x):
            return 0.5 + ((bm.sin(bm.linalg.norm(x))) ** 2 - 0.5) / ((1 + 0.001 * (bm.linalg.norm(x) ** 2)) ** 2)

        def F20(x):
            return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4

        def F21(x):
            return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[0]) * bm.cos(4 * bm.pi * x[1]) + 0.3

        def F22(x):
            return x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * bm.cos(3 * bm.pi * x[0] + 4 * bm.pi * x[1]) + 0.3

        def F23(x):
            x1 = x[0]
            x2 = x[1]
            sum1 = 0
            sum2 = 0
            for i in range(1, 6):
                sum1 += i * bm.cos((i + 1) * x1 + i)
                sum2 += i * bm.cos((i + 1) * x2 + i)
            return sum1 * sum2

        def F24(x):
            o = 0
            n = len(x)
            for i in range(n - 1):
                o += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
            return o

        def F25(x):
            n = len(x)
            y1 = bm.sum(x ** 2) / 4000
            y2 = 1
            for i in range(n):
                y2 *= bm.cos(x[i] / bm.sqrt(i + 1))
            return 1 + y1 - y2

        def F26(x):
            y1 = bm.sum(x ** 2)
            y2 = bm.sum(bm.cos(2 * bm.pi * x))
            n = len(x)
            return -20 * bm.exp(-0.2 * bm.sqrt(y1 / n)) - bm.exp(y2 / n) + 20 + bm.exp(1)

        
        Functions_dict = {
            'F1' : (F1, -5.12, 5.12, 30),
            'F2' : (F2, -100, 100, 30),
            'F3' : (F3, -10, 10, 30),
            'F4' : (F4, -100, 100, 30),
            'F5' : (F5, -4.5, 4.5, 2),
            'F6' : (F6, -100, 100, 2),
            'F7' : (F7, -10, 10, 2),
            'F8' : (F8, -10, 10, 4),
            'F9' : (F9, -5, 10, 10),
            'F10' : (F10, -10, 10, 30),
            'F11' : (F11, -10, 10, 10),
            'F12' : (F12, -10, 10, 30),
            'F13' : (F13, -100, 100, 2),
            'F14' : (F14, -10, 10, 2),
            'F15' : (F15, -bm.pi, bm.pi, 2),
            'F16' : (F16, -bm.pi, bm.pi, 5),
            'F17' : (F17, -bm.pi, bm.pi, 10),
            'F18' : (F18, -5.12, 5.12, 30),
            'F19' : (F19, -100, 100, 2),
            'F20' : (F20, -5, 5, 2),
            'F21' : (F21, -100, 100, 2),
            'F22' : (F22, -100, 100, 2),
            'F23' : (F23, -10, 10, 5),
            'F24' : (F24, -30, 30, 30),
            'F25' : (F25, -600, 600, 30),
            'F26' : (F26, -32, 32, 30),
        }

        # 检查case是否在字典的键中  
        if self.F in Functions_dict:  
            fobj, lb, ub, dim, = Functions_dict[self.F] 

            print(f"Function: {fobj.__name__}, Lower Bound: {lb}, Upper Bound: {ub}, Dimension: {dim}")  
        else:  
            print("No matching case found.")  
        
        return fobj, lb, ub, dim



