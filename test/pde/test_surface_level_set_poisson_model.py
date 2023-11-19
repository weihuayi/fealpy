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
import ipdb


x, y, z = sp.symbols('x, y, z', real=True)

F1 = x**2 + y**2 + z**2 - 1 
u1 = x * y
pde1 = SurfaceLevelSetPDEData(F1, u1)


#F2 = x**2/81 + y**2/9 + z**2 - 1
#u2 = np.sin(x)
#pde2 = SurfaceLevelSetPDEData(F2, u2)

def test_surface_level_set_poisson_model1():
    p = np.array([[0, 1, 0]], dtype=np.float64)
    print(p.shape)

    assert np.allclose(pde1.levelset(p), 0, atol=1e-9)
    assert np.allclose(pde1.solution(p), p[:, 0]*p[:, 1], atol=1e-9) 
    
    nabla = np.array([[p[:, 1]], [p[:, 0]], [np.zeros(p.shape[0])]]) - 2*p[:, 0]*p[:, 1]/(p[:, 0]**2+p[:, 1]**2+p[:, 2]**2)*np.array([[p[:, 0]], [p[:, 1]], [p[:, 2]]])
    
    #ipdb.set_trace()
    assert np.allclose(pde1.graddient(p), nabla, atol=1e-9)

    #ff = (4*p[0]**3*p[1]+4*p[0]*p[1]**3+4*p[0]*p[1]*p[2]**2)/((p[0]**2+p[1]**2+p[2]**2)**2) - 10*p[0]*p[1]/(p[0]**2+p[1]**2+p[2]**2) 
    #print(ff)
    #print(pde1.source(p))
    #assert np.allclose(pde1.source(p), ff, atol=1e-9)
'''
def test_surface_level_set_poisson_model2():
    p = np.array([0, 1, 0], dtype=np.float64)

    assert np.allclose(pde2.levelset(p), 0, atol=1e-9)
    assert np.alclose(pde2.solution(p), np.sin(x), atol=1e-9)

    nabla = 
    assert np.allclose(pde2.graddient(p), nabla, atol=1e-9)

    ff = 
    assert np.allclose(pde2.source(p), ff, atol=1e-9)
'''

if __name__ == "__main__":
    test_surface_level_set_poisson_model1()
    #test_surface_level_set_poisson_model2()
