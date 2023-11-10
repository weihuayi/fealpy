#!/usr/bin/python3
'''!    	
	@Author: lq
	@File Name: test_surface_level_set_poisson_model.py
	@Mail: 2655493031@qq.com 
	@Created Time: 2023年11月08日 星期三 19时36分12秒
	@bref 
	@ref 
'''  
import pytest
import numpy as np
import sympy as sp

def test_surface_level_set_poisson_model():
    from fealpy.pde.surface_level_set_poisson_model import SurfaceLevelSetPDEData

    x, y, z = sp.symbols('x, y, z', real=True)

    F = x**2 + y**2 + z**2 -1
    u = x * y

    pde = SurfaceLevelSetPDEData(F, u)



if __name__ == "__main__":

    test_surface_level_set_poisson_model()
