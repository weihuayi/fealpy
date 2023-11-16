#!/usr/bin/python3
'''!    	
	@Author: lq
	@File Name: test_surface_level_set_poisson_model.py
	@Mail: 2655493031@qq.com 
	@Created Time: 2023年11月08日 星期三 19时36分12秒
	@bref 
	@ref 
'''  
import numpy as np
import sympy as sp
import pytest
from fealpy.pde.surface_level_set_poisson_model import SurfaceLevelSetPDEData

x, y, z = sp.symbols('x, y, z', real=True)

F = x**2 + y**2 + z**2 -1
u = x * y

pde = SurfaceLevelSetPDEData(F, u)

def test_surface_level_set_poisson_model():
    p = np.array([0, 1, 0], dtype=np.float64)

    assert np.allclose(pde.levelset(p), 0, atol=1e-9)
    assert np.allclose(pde.solution(p), p[0]*p[1], atol=1e-9) 

    nabla = np.array([p[1], p[0], 0]) - 2*p[0]*p[1]/(p[0]**2+p[1]**2+p[2]**2)*np.array([p[0], p[1], p[2]])
    #nabla = pde.udiff(p[:, 0], p[:, 1], p[:, 2])
    assert np.allclose(pde.graddient(p), nabla.reshape(-1,1), atol=1e-9)

    ff = (4*p[0]**3*p[1]+4*p[0]*p[1]**3+4*p[0]*p[1]*p[2]**2)/((p[0]**2+p[1]**2+p[2]**2)**2) - 10*p[0]*p[1]/(p[0]**2+p[1]**2+p[2]**2) 
    assert np.allclose(pde.source(p), ff, atol=1e-9)

if __name__ == "__main__":
    test_surface_level_set_poisson_model()

