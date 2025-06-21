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

def test_surface_level_set_poisson_model1():
    F1 = x**2 + y**2 + z**2 - 1 
    u1 = x * y
    pde1 = SurfaceLevelSetPDEData(F1, u1)

    p = np.array([[0, 1, 0]], dtype=np.float64)

    assert np.allclose(pde1.levelset(p), 0, atol=1e-9)
    assert np.allclose(pde1.solution(p), p[:, 0]*p[:, 1], atol=1e-9) 
    
    #ipdb.set_trace()
    nabla_u = np.array([[p[:, 1]], [p[:, 0]], [np.zeros(p.shape[0])]]) 
    nu_n = 2*p[:, 0]*p[:, 1]/(p[:, 0]**2+p[:, 1]**2+p[:, 2]**2)*np.array([[p[:, 0]], [p[:, 1]], [p[:, 2]]])
    nabla = nabla_u - nu_n
    assert np.allclose(pde1.graddient(p), nabla, atol=1e-9)

    ff = -4*p[:, 0]*p[:, 1] / (p[:, 0]**2+p[:, 1]**2+p[:, 2]**2)
    assert np.allclose(pde1.source(p), ff, atol=1e-9)


def test_surface_level_set_poisson_model2():
    F2 = x**2/81 + y**2/9 + z**2 - 1
    u2 = sp.sin(x)
    pde2 = SurfaceLevelSetPDEData(F2, u2)

    p = np.array([[9, 0, 0]], dtype=np.float64)

    #ipdb.set_trace()
    assert np.allclose(pde2.levelset(p), 0, atol=1e-9)
    assert np.allclose(pde2.solution(p), np.sin(p[:, 0]), atol=1e-9)

    nabla_u = np.array([[np.cos(p[:, 0])], [np.zeros(p.shape[0])], [np.zeros(p.shape[0])]])
    nu_n = p[:, 0]*np.cos(p[:, 0])/(81*(p[:, 0]**2/9**4+p[:, 1]/3**4+p[:, 2]**2))*np.array([[p[:, 0]/9**2], [p[:, 1]/3**2], [p[:, 2]]])
    nabla = nabla_u - nu_n
    assert np.allclose(pde2.graddient(p), nabla, atol=1e-9)

    nu_nn = p[:, 0]*np.cos(p[:, 0])*(10*p[:, 0]**2/9**2+82*p[:, 1]**2/3**2+90*p[:, 2]**2) / (9**5*(p[:, 0]**2/9**4+p[:, 1]**2/3**4+p[:, 2]**2)**2) 
    n_hu = p[:, 0]**2*np.sin(p[:, 0]) / (9**4*(p[:, 0]**2/9**4+p[:, 1]**2/3**4+p[:, 2]**2))
    ff = -np.sin(p[:, 0]) - nu_nn + n_hu
    assert np.allclose(pde2.source(p), ff, atol=1e-9)

if __name__ == "__main__":
    test_surface_level_set_poisson_model1()
    test_surface_level_set_poisson_model2()
