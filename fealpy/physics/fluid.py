
import numpy as np



class Fluid():
    def __init__(self):
        pass

class Parameter():
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit


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

