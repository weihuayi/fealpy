import numpy as np

class Parameter():
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

class Constant():
    R = Parameter(8.31446261815324, 'J/K/mol')
    g = Parameter(9.80665, 'm/s^2')
