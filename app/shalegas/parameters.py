
import numpy as np

"""
Wei Yu and so on. CO2 injection for enhanced oil recovery in Bakken tight oil reservoirs, 2015


1 bar = 100,000 Pa 

TODO
----
1. 添加 H2O 和 CH4 的相互作用参数


Reference
---------
http://www.coolprop.org/fluid_properties/fluids/Methane.html?highlight=ch4

"""

class Substance:
    def __init__(self):
        self.R = 8.31446261815324 # 理想气体常数 J/K/mol
        n=5
        self.substance = {
                'CH4':   0, 
                'C2H6':  1, 
                'C3H8':  2, 
                'C4H10': 3, 
                'C5H12': 5,
                'H2O':   n+1,
                'CO2':   n+2,
                'N2':    n+3}

        #组分属性:
# 0: Molar mass(kg/mol)  1:T_c(K)  2:P_c(MPa)  3:rho_c(kg/m^3) 4: rho_c(mol/m^3) 5:V_c(m^3/kg) 6: Z_c 7: T_b(K)  8: Omega   9: Shift parameter
        self.cproperty = np.array([
            (0.016042800, 190.564, 4.599200, 162.6600026784, 10139.128, 6.14614e-3, 0.288, 111.63, 0.01142, -0.15400),   # CH4
            (305.42, 4.872, 4.83881e-3, 0.285, 184.55, 0.099, -0.10020),   # C2H6
            (369.83, 4.248, 4.53554e-3, 0.281, 231.01, 0.153, -0.08501),   # C3H8
            (407.80, 3.604, 4.45607e-3, 0.274, 272.64, 0.183, -0.07935),   # C4H10 
            (460.40, 3.380, 4.42118e-3, 0.271, 309.22, 0.227, -0.04350),   # C5H12
            (0.018015268, 647.096, 22.064, 2.50000e-3, 0.229, 373.30, 0.3442920843,  0000000),   # H2O
            (304.10, 0.380, 2.10000e-3, 0.274, 194.80, 0.240,  0.06000),   # CO2
            (126.20, 3.390, 3.20000e-3, 0.290,  77.50, 0.040, -0.28900),   # N2
            ], dtype=np.float64)

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


