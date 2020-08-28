
import numpy as np


substance = {'CH4': 0, 'C2H6': 1, 'C3H8':2, 'H2O':3}

R = 8.31446261815324 # 理想气体常数 J/K/mol

#组分属性:
# 0:T_c(K) 1:P_c(MPa)  2:rho_c(g/cm^3) 3: T_b(K)  4: Z_c 
cproperty = np.array([
    (190.58, 4.604, 0.162, 111.63, 0.288),
    (305.42, 4.880, 0.203, 184.55, 0.285),
    (369.82, 4.250, 0.217, 231.05, 0.281),
    (647.4, 22.104, 0.400, 373.30, 0.229)], dtype=np.float64)


"""
Wei Yu and so on. CO2 injection for enhanced oil recovery in Bakken tight oil reservoirs, 2015

TODO
----
1. H2O 和 CH4 的相互作用参数
"""
# 组分相互作用参数
cinteraction = np.array([
    (0, 0.0078, 0.0078, 0),
    (0.0078, 0, 0.0078, 0),
    (0.0078, 0.0078, 0, 0),
    (0, 0, 0, 0)], dtype=np.float64)


