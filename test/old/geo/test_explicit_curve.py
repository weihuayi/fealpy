
import numpy as np
import matplotlib.pyplot as plt
import pytest

def test_lagrange_curve():
    pass


def test_bspline_curve():
    from fealpy.geometry import BSplineCurve
    n = 9
    p = 2
    knot = np.array([
        0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.8, 1, 1, 1
        ], dtype=np.float64)
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40, 60],
        [30, 90]], dtype=np.float64)
    

    curve = BSplineCurve(n, p, knot, node)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')
    
    xi = np.linspace(0, 1, 1000)
    bval = curve.basis(xi)
    fig = plt.figure()
    axes = fig.gca()
    axes.plot(xi, bval)
    plt.show()

def test_crspline_curve():
    from fealpy.geometry import CRSplineCurve
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40,60]], dtype=np.float64)

    c0 = CRSplineCurve(node, 0.2)
    c1 = CRSplineCurve(node, 0.5)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')

    xi = np.linspace(0, 1, 1000)
    ps0 = c0(xi)
    ps1 = c1(xi)
    axes.plot(ps0[:, 0], ps1[:, 1], 'r', ps1[:, 0], ps1[:, 1], 'k')
    plt.show()

def test_chspline_curve():
    node = np.array([
        [20, 5], 
        [10, 20], 
        [40, 50], 
        [60, 5], 
        [70, 8], 
        [100,56],
        [50, 50], 
        [40,60]], dtype=np.float64)


    tang0 = np.zeros_like(node)
    tang0[1:-1] = node[2:] - node[0:-2]
    tang0[0] = 2*node[1] - 2*node[0]
    tang0[-1] = 2*node[-1] - 2*node[-2]
    c0 = CHSplineCurve(node, 0.2*tang0)
    c1 = CHSplineCurve(node, 0.5*tang0)

    fig = plt.figure()
    axes = fig.gca()
    axes.plot(node[:, 0], node[:, 1], 'b-.')

    xi = np.linspace(0, 1, 1000)
    ps0 = c0(xi)
    ps1 = c1(xi)
    axes.plot(ps0[:, 0], ps1[:, 1], 'r', ps1[:, 0], ps1[:, 1], 'k')
    plt.show()

if __name__ == '__main__':
    test_bspline_curve()
    test_crspline_curve()
    test_chspline_curve()
