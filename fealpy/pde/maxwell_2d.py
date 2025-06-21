
from ..backend import backend_manager as bm
from ..mesh import TriangleMesh
from ..decorator import cartesian, barycentric


class SinData():
    def __init__(self, eps = 1, k = 1):
        """
        @brief 求解方程 
                curl curl E - eps E = J     Omega
                          n \times E = g     Gamma0
             - curl E - k n \times E = f     Gamma1
        """
        self.eps = eps
        self.k = k

    @cartesian
    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        Fx = x*y*(1 - x)*(1 - y)
        Fy = bm.sin(bm.pi*x)*bm.sin(bm.pi*y)
        #Fx=2*x+y
        #Fy=5*x+3 
        f = bm.concatenate([Fx, Fy], axis=-1) 
        return f 

    @cartesian
    def curl_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos

        curlF = -x*y*(x - 1) - x*(1 - x)*(1 - y) + pi*sin(pi*y)*cos(pi*x)
        # curlF = 4
        return curlF

    @cartesian
    def curl_curl_solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        pi = bm.pi
        sin = bm.sin
        cos = bm.cos

        ccFx = 2*x*(1 - x) + pi**2*cos(pi*x)*cos(pi*y)
        ccFy = x*y - x*(1 - y) - y*(1 - x) - (1 - y)*(x - 1) + pi**2*sin(pi*x)*sin(pi*y)
        # ccFx=0*x
        # ccFy=0*y
        ccf = bm.concatenate([ccFx, ccFy] , axis=-1)
        return ccf

    @cartesian
    def source(self, p):
        return self.curl_curl_solution(p) - self.eps * self.solution(p)

    @cartesian
    def dirichlet(self, p, t):
        val = self.solution(p)
        if t.ndim == 2:
            return bm.einsum('eqd, ed->eq', val, t)
        else:
            return bm.einsum("eqd,eqd->eq", val, t)
    


    @cartesian
    def neumann(self, p, t):
        """!
        @param p: (..., N, ldof, 3)
        @param t: (N, 3)
        """
        t = t[:, None, :]
        return t*self.curl_solution(p)[..., None]

    def robin(self, p, n):
        """!
        @param p: (..., N, ldof, 3)
        @param n: (N, 3)
        """
        cf = self.curl_solution(p)
        fval = self.solution(p)
        return -cf - np.cross(n[None, :], fval)

    def boundary_type(self, mesh):
        bdface = mesh.ds.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        isLeftBd   = np.abs(f2n[:, 0]+1)<1e-14
        isRightBd  = np.abs(f2n[:, 0]-1)<1e-14
        isBottomBd = np.abs(f2n[:, 1]+1)<1e-14
        isUpBd     = np.abs(f2n[:, 1]-1)<1e-14
        bd = {"left": bdface[isLeftBd], "right": bdface[isRightBd], 
              "up": bdface[isUpBd], "bottom": bdface[isBottomBd], }
        return bd

    def init_mesh(self, nx=1, ny=1, meshtype='tri'):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=nx, ny=ny)
        return mesh

    def domain(self):
        return [0, 1, 0, 1]

