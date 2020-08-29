
import numpy as np

"""
Wei Yu and so on. CO2 injection for enhanced oil recovery in Bakken tight oil reservoirs, 2015


1 bar = 100,000 Pa 

TODO
----
1. 添加 H2O 和 CH4 的相互作用参数
"""

class Substance:
    def __init__(self):
        self.R = 8.31446261815324 # 理想气体常数 J/K/mol
        self.substance = {
                'CH4':   0, 
                'C2H6':  1, 
                'C3H8':  2, 
                'C4H10': 3, 
                'C5H12': 5,
                'H2O':   6}

        #组分属性:
# 0:T_c(K) 1:P_c(MPa)  2:V_c(m^3/kg)   3: Z_c      4: T_b(K)    5: Omega   6: Shift parameter
# 临界温度  临界压强     临界体积    临界压缩因子   沸点温度    偏心因子 
        self.cproperty = np.array([
            (190.56, 4.599, 6.14614e-3, 111.63, 0.288, 0.011, -0.154),   # CH4
            (305.42, 4.872, 0.203, 184.55, 0.285, 0.099, -0.1002),  # C2H6
            (369.83, 4.248, 0.217, 231.05, 0.281, 0.153, -0.08501), # C3H8
            (407.80, 3.604,   0.0,    0.0,   0.0,   0.0,    0.0),   # C4H10 
            (),                                                     # C5H12
            (647.4, 22.104, 0.400, 373.30, 0.229)]                  # H2O
            , dtype=np.float64)

        # 混合物组分相互作用参数
        self.cinteraction = np.array([
            (     0, 0.0078, 0.0078, 0),
            (0.0078,      0, 0.0078, 0),
            (0.0078, 0.0078,      0, 0),
            (     0,      0,      0, 0)], dtype=np.float64)

    def acentric_factor(self, s):
        """

        Notes
        -----
        计算一组不同物质的偏心因子
        """
        n = len(s) # 物质的个数
        index = np.zeros(n, dtype=np.int_)
        for i, name in enumerate(s):
            index[i] = self.substance[name]

        c = 3.0/7.0
        pc = self.cproperty[index, 1]*1.45037737730e+2 # 临界压强, MPa to PISA
        Tc = self.cproperty[index, 0] # 临界温度
        Tb = self.cproperty[index, 3] # 沸点温度
        Tr = Tc/Tb
        omega = c*np.log(pc/14.695)/(Tr-1) - 1
        return omega

    def alpha_factor(self, s):
        """

        Notes
        -----

        """
        n = len(s) # 物质的个数
        index = np.zeros(n, dtype=np.int_)
        for i, name in enumerate(s):
            index[i] = self.substance[name]

        c = 3.0/7.0
        pc = self.cproperty[index, 1]*1.45037737730e+2 # 临界压强, MPa to PISA
        Tc = self.cproperty[index, 0] # 临界温度
        Tb = self.cproperty[index, 3] # 沸点温度
        Tr = Tc/Tb
        omega = c*np.log(pc/14.695)/(Tr-1) - 1
        
        print(omega)
        m = 0.37464 + 1.54226*omega - 0.26992*omega**2 # 0< \omega < 0.5
        # m = 0.3796 + 1.485*omega-0.1644*omega**2+0.01667*omega**3 # 0.1 < omega < 2
        alpha = (1 + m*(1 - np.sqrt(Tr)))**2

    def compressibility_factor(self, s, p, T, c):
        """
        Notes
        -----
        计算混合物的压缩因子
        s: 物质类型
        p: 总的压强
        T: 环境温度
        c: 摩尔分数
        """

        R  = self.R

        n = len(s) # 物质的个数
        index = np.zeros(n, dtype=np.int_)
        for i, name in enumerate(s):
            index[i] = self.substance[name]

        t = 3.0/7.0
        pc = self.cproperty[index, 1]*1.45037737730e+2 # 临界压强, MPa to PISA
        Tc = self.cproperty[index, 0] # 临界温度
        Tb = self.cproperty[index, 3] # 沸点温度
        Tr = Tc/Tb
        omega = t*np.log(pc/14.695)/(Tr-1) - 1
        
        print(omega)
        #m = 0.37464 + 1.54226*omega + 0.26992*omega**2 # 0< \omega < 0.5
        m = 0.3796 + 1.485*omega - 0.1644*omega**2 + 0.01667*omega**3 # 0.1 < omega < 2
        alpha = (1 + m*(1 - np.sqrt(Tr)))**2

        ac = 0.45724*R**2*Tc**2/pc
        bc = 0.07780*R*Tc/pc

        a = ac*alpha
        b = bc # TODO: 确认这里的计算过程

        sa = np.sqrt(a)
        I = np.sqrt((sa[:, None]*sa)*(1 - self.cinteraction[index, :][:, index]))
        a = np.sum((I@c)*c)
        b = np.sum(bc*c)

        A = a*p/R**2/T**2
        B = b*p/R/T

        t = np.ones(4, dtype=np.float64)
        t[1] = B - 1
        t[2] = A - 3*B**2 - 2*B
        t[3] = -A*B + B**2 + B**3
        Z = np.roots(t)
        print(Z)


if __name__ == '__main__':
    substance = Substance()
    s = ['CH4', 'C2H6', 'C3H8']
    p = 5e6 # Pa 
    T = 397 # K
    c = np.array([1, 0, 0])
    substance.compressibility_factor(s, p, T, c)


