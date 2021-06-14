from fealpy.quadrature import FEMMeshIntegralAlg
integralalg = FEMMeshIntegralAlg(mesh, 3)
integralalg.mesh_integral(u, q=3, power=2)
integralalg.error(u, uh, q=3, power=2)



from fealpy.quadrature import TriangleQuadrature

qf = TriangleQuadrature(3)
bcs, ws = qf.get_quadrature_points_and_weights()
