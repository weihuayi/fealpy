from abc import ABC, abstractmethod

class BaseEquation(ABC):
    """ Base class for simulation """
    def __init__(self, pde):
        self.pde = pde
    
    @property
    def coefs(self):
        """ Coefficients for equation  """
        pass

    @property
    def variables(self):
        """ Variables for equation """
        pass
    
    def __str__(self) -> str:
        """ String representation of the equation """
        pass
