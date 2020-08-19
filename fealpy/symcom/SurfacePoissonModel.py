
from sympy import * 
from sympy.vector import CoordSys3D, gradient, divergence
from sympy.tensor.array import derive_by_array
from sympy.abc import x, y, z

class SurfacePoissonModel:

    def __init__(self, u, phi):
        self.u = u
        self.phi = phi

    def show_surface(self):
        phi = self.phi
        n0 = derive_by_array(phi, (x, y, z))
        n /= sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        hn = derive_by_array(n, (x, y, z)).tomatrix()
        dn = tensorcontraction(hn, (0, 1))

    def show_pde(self):

        u = self.u
        phi = self.phi

        gu = derive_by_array(u, (x, y, z))
        hu = derive_by_array(gu, (x, y, z)).tomatrix()
        lu = tensorcontraction(hu, (0, 1))

        n = derive_by_array(phi, (x, y, z))
        n /= sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        hn = derive_by_array(n, (x, y, z)).tomatrix()
        dn = tensorcontraction(hn, (0, 1))

        gu = Matrix(gu)
        n = Matrix(n)

        t0 = lu - (gu.transpose()*n)[0, 0]*dn 
        t1 = n.transpose()*hu*n 
        f = -t0 + t1[0, 0] 

        gu -= (gu.T*n)[0, 0]*n

        return u, gu, f.simplify() 


        


if __name__ == "__main__":
    print("Model 0:")

    u = sin(pi*x)*sin(pi*y)*sin(pi*z) 
    phi = sqrt(x**2 + y**2 + z**2) - 1
    model = SurfacePoissonModel(u,phi) 

    u, gu, f = model.show_pde()
    print('u', u)
    print('\n gu', gu)
    print('\n f', f)
