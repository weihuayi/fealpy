#!/usr/bin/python3
'''!    	
	@Author: lq
	@File Name: test_sphere_surface_poisson_model.py
	@Mail: 2655493031@qq.com
	@Created Time: 2023年10月26日 星期四 19时43分12秒
	@bref 
	@ref 
'''  
import pytest
import numpy as np
import sympy as sp

def test_sphere_surface_poisson_model():
    from fealpy.pde.sphere_surface_poisson_model_3d import SphereSurfacePDEData 

    x, y, z = sp.symbols('x, y, z', real=True)

    F = x**2 + y**2 +z**2-1
    u = x *y

    pde = SphereSurfacePDEData(F, u)
    print("f:", f)

    p = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=np.float64)
    
    print("solution:", pde.solution(p))
    print("graddient:", pde.graddient(p))
    print("source:", pde.source(p))



if __name__ == "__main__":

    test_sphere_surface_poisson_model()
