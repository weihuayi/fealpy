
"""

Reference
---------
1. http://www.coolprop.org/


Template
--------

class T(Fluid):
    molar_mass = Parameter(, 'kg/mol')
    acentric_factor = Parameter(, None)
    formula = ''

    maximum_temperature = Parameter(, 'K')
    maximum_pressure = Parameter(, 'Pa')

    triple_point_temperature = Parameter(, 'K')
    triple_point_pressure = Parameter(, 'Pa')

    critical_point_temperature = Parameter(, 'K')
    critical_point_density  = Parameter(, 'kg/m^3')
    critical_point_pressure = Parameter(, 'Pa')

    boiling_temperature = Parameter(, 'K') 
"""

import numpy as np

from .constant import Parameter, Constant

class Fluid():
    @classmethod
    def critical_compressibility_factor(cls):
        """
        Notes
        -----
        临界点处流体的压缩因子
        """

        R = Constant.R.value
        omega = cls.acentric_factor.value 
        m = 0.37464 + 1.54226*omega - 0.26992*omega**2 # 0< \omega < 0.5

        Pc = cls.critical_point_pressure.value
        Tc = cls.critical_point_temperature.value
        a = 0.45724*R**2*Tc**2/Pc
        b = 0.07780*R*Tc/Pc

        A = a*Pc/R**2/Tc**2
        B = b*Pc/R/Tc

        t = np.ones(4, dtype=np.float64)
        t[1] = B - 1
        t[2] = A - 3*B**2 - 2*B
        t[3] = -A*B + B**2 + B**3
        Z = np.roots(t)
        print(Z)

    @classmethod
    def compressibility_factor(cls, p, T):
        """
        """
        R = Constant.R.value
        omega = cls.acentric_factor.value 
        k = 0.37464 + 1.54226*omega - 0.26992*omega**2 # 0< \omega < 0.5

        Pc = cls.critical_point_pressure.value
        Tc = cls.critical_point_temperature.value
        Tb = cls.boiling_temperature.value
        Tr = Tc/Tb
        ac = 0.45724*R**2*Tc**2/Pc
        bc = 0.07780*R*Tc/Pc

        alpha = (1 + k*(1 - Tr))**2

        a = ac*alpha
        b = bc

        A = a*p/R**2/T**2
        B = b*p/R/T

        t = np.ones(4, dtype=np.float64)
        t[1] = B - 1
        t[2] = A - 3*B**2 - 2*B
        t[3] = -A*B + B**2 + B**3
        Z = np.roots(t)
        print(Z)

# 氧气, O2
class Oxygen(Fluid):
    molar_mass = Parameter(0.0319988, 'kg/mol')
    acentric_factor = Parameter(0.0222, None)
    formula = 'O2'

    maximum_temperature = Parameter(2000.0, 'K')
    maximum_pressure = Parameter(80000000.0, 'Pa')

    triple_point_temperature = Parameter(54.361, 'K')
    triple_point_pressure = Parameter(146.27764705809653, 'Pa')

    critical_point_temperature = Parameter(154.58100000000002, 'K')
    critical_point_density  = Parameter(436.143644, 'kg/m^3')
    critical_point_pressure = Parameter(5043000.0, 'Pa')

    boiling_temperature = Parameter(1.0, 'K') # TODO: make correct

# 氮气, N2
class Nitrogen(Fluid):
    molar_mass = Parameter(0.02801348, 'kg/mol')
    acentric_factor = Parameter(0.0372, None)
    formula = 'N2'

    maximum_temperature = Parameter(2000.0, 'K')
    maximum_pressure = Parameter(2200000000.0, 'Pa')

    triple_point_temperature = Parameter(63.151, 'K')
    triple_point_pressure = Parameter(12519.78348430944, 'Pa')

    critical_point_temperature = Parameter(126.192, 'K')
    critical_point_density  = Parameter(313.3, 'kg/m^3')
    critical_point_pressure = Parameter(3395800.0, 'Pa')

    boiling_temperature = Parameter(77.50, 'K')

# 二氧化碳, CO2 
class CarbonDioxide(Fluid):
    molar_mass = Parameter(0.0440098, 'kg/mol')
    acentric_factor = Parameter(0.22394, None)
    formula = 'CO2'

    maximum_temperature = Parameter(2000.0, 'K')
    maximum_pressure = Parameter(800000000.0, 'Pa')

    triple_point_temperature = Parameter(216.592, 'K')
    triple_point_pressure = Parameter(517964.34344772575, 'Pa')

    critical_point_temperature = Parameter(304.1282, 'K')
    critical_point_density  = Parameter(467.60000128174005, 'kg/m^3')
    critical_point_pressure = Parameter(7377300.0, 'Pa')

    boiling_temperature = Parameter(194.80, 'K')

# 水, H2O
class Water(Fluid):
    molar_mass = Parameter(0.018015268, 'kg/mol')
    acentric_factor = Parameter(0.3442920843, None)
    formula = 'H2O'

    maximum_temperature = Parameter(2000.0, 'K')
    maximum_pressure = Parameter(1000000000.0, 'Pa')

    triple_point_temperature = Parameter(273.16, 'K')
    triple_point_pressure = Parameter(611.6548008968684, 'Pa')

    critical_point_temperature = Parameter(647.096, 'K')
    critical_point_density  = Parameter(322.00000000000006, 'kg/m^3')
    critical_point_pressure = Parameter(22064000.0, 'Pa')

    boiling_temperature = Parameter(373.30, 'K')

# 甲烷，CH4
class Methane(Fluid):
    molar_mass = Parameter(0.0160428, 'kg/mol')
    acentric_factor = Parameter(0.01142, None)
    formula = 'CH4'

    maximum_temperature = Parameter(625.0, 'K')
    maximum_pressure = Parameter(1000000000.0, 'Pa')

    triple_point_temperature = Parameter(90.6941, 'K')
    triple_point_pressure = Parameter(11696.064114962215, 'Pa')

    critical_point_temperature = Parameter(190.564, 'K')
    critical_point_density  = Parameter(162.6600026784, 'kg/m^3')
    critical_point_pressure = Parameter(4599200.0, 'Pa')

    boiling_temperature = Parameter(111.63, 'K')


# 乙烷，C2H6
class Ethane(Fluid):
    molar_mass = Parameter(0.03006904, 'kg/mol')
    acentric_factor = Parameter(0.099, None)
    formula = 'C2H6'

    maximum_temperature = Parameter(675.0, 'K')
    maximum_pressure = Parameter(900000000.0, 'Pa')

    triple_point_temperature = Parameter(90.368, 'K')
    triple_point_pressure = Parameter(1.142107639085233, 'Pa')

    critical_point_temperature = Parameter(305.322, 'K')
    critical_point_density  = Parameter(206.18000000673237, 'kg/m^3')
    critical_point_pressure = Parameter(4872200.0, 'Pa')

    boiling_temperature = Parameter(184.55, 'K')

# 丙烷，C3H8
class Propane(Fluid):
    molar_mass = Parameter(0.04409562, 'kg/mol')
    acentric_factor = Parameter(0.1521, None)
    formula = 'C3H8'

    maximum_temperature = Parameter(650.0, 'K')
    maximum_pressure = Parameter(1000000000.0, 'Pa')

    triple_point_temperature = Parameter(85.525, 'K')
    triple_point_pressure = Parameter(0.00017184840809308612, 'Pa')

    critical_point_temperature = Parameter(369.89, 'K')
    critical_point_density  = Parameter(220.47810000000004, 'kg/m^3')
    critical_point_pressure = Parameter(4251200.0, 'Pa')

    boiling_temperature = Parameter(231.05, 'K')
