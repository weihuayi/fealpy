
from numpy.typing import NDArray
from typing import TypedDict, Callable, Tuple, Union

import numpy as np


class ScaledMonomialSpace2dScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray]):
        """
        @brief

        @param[in] f 
        """
        self.f = f

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None,
            out=None, q=None):

        pass

class ScaledMonomialSpace3dScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray]):
        """
        @brief

        @param[in] f 
        """
        self.f = f

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None,
            out=None, q=None):

        pass

class ScaledMonomialSpace2dVectorSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray]):
        """
        @brief

        @param[in] f 
        """
        self.f = f

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None,
            out=None, q=None):

        pass

class ScaledMonomialSpace3dVectorSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray]):
        """
        @brief

        @param[in] f 
        """
        self.f = f

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None,
            out=None, q=None):

        pass
