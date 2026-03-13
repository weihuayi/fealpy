
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MicrostripAntenna3D"]


class MicrostripAntenna3D(CNodeType):
    TITLE: str = "三维微带贴片天线"
    PATH: str = "模型.双旋度"
    DESC: str = "三维微带贴片天线节点"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("space", DataType.MENU, 0, title="函数空间类型", items=["First Nédélec"]),
        PortConf("p", DataType.INT, 1, title="Nédélec 阶数", default=1),
        PortConf("f", DataType.FLOAT, 1, title="频率(GHz)", default=1.575),
        PortConf("sub_region", DataType.TENSOR, 1, title="基板单元"),
        PortConf("air_region", DataType.TENSOR, 1, title="空气单元"),
        PortConf("pec_face", DataType.TENSOR, 1, title="PEC 边界"),
        PortConf("lumped_edge", DataType.TENSOR, 1, title="集总端口"),
        PortConf("mu_sub", DataType.FLOAT, 1, title="基板相对磁导率", default=1.0),
        PortConf("epsilon_sub", DataType.FLOAT, 1, title="基板相对介电常数", default=3.38),
        PortConf("mu_air", DataType.FLOAT, 1, title="空气相对磁导率", default=1.0),
        PortConf("epsilon_air", DataType.FLOAT, 1, title="空气相对介电常数", default=1.0),
        PortConf("r0", DataType.FLOAT, 1, title="PML 内径(mm)", default=100.0),
        PortConf("r1", DataType.FLOAT, 1, title="PML 外径(mm)", default=120.0),
        PortConf("s", DataType.FLOAT, 1, title="PML 多项式系数", default=5.0),
        PortConf("pp", DataType.FLOAT, 1, title="PML 多项式次数", default=2.0)
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.FUNCTION, title="算子"),
        PortConf("vector", DataType.FUNCTION, title="向量"),
        PortConf("uh", DataType.FUNCTION, title="初值")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.model.curlcurl.curlcurl_sphere_pml import CurlCurlSpherePML
        from fealpy.functionspace import FirstNedelecFESpace
        from fealpy.decorator import barycentric, cartesian

        pml = CurlCurlSpherePML(
            r0=options['r0'],
            r1=options['r1'],
            omega=2*bm.pi*options['f']*1e9,
            mu=options['mu_air'],
            epsilon=options['epsilon_air'],
            s=options['s'],
            p=options['pp'],
        ) # PML is in the air
        k = 2*bm.pi*options['f']*1e1 / 3
        mesh = options['mesh']
        p = options['p']
        space = FirstNedelecFESpace(mesh, p)
        ALL = slice(None)

        from ...fem import (
            BilinearForm,
            LinearForm,
            ScalarMassIntegrator,
            CurlCurlIntegrator,
            ScalarSourceIntegrator
        )
        MSA = MicrostripAntenna3D

        @barycentric
        def diffusion(bcs, index):
            pp      = mesh.bc_to_point(bcs)
            r       = bm.linalg.norm(pp, axis=-1)
            isInPML = r > options['r0']
            F       = pml.jacobi(pp[isInPML])
            detF    = pml.detjacobi(pp[isInPML])

            NC = mesh.number_of_cells()
            NQ = bcs.shape[0]
            is_air, is_sub = options["air_region"], options["sub_region"]
            mu_air, mu_sub = options["mu_air"], options["mu_sub"]

            val = bm.zeros((NC, NQ, 3, 3), dtype=bm.complex128, device=mesh.device)
            for i in range(3):
                val = bm.set_at(val, (is_air, ALL, i, i), 1/mu_air)
                val = bm.set_at(val, (is_sub, ALL, i, i), 1/mu_sub)

            FTF = bm.einsum("...ij, ...ik -> ...jk", F, F)
            val = bm.set_at(val, isInPML, FTF / detF[..., None, None] / mu_air)

            return val

        @barycentric
        def reaction(bcs, index):
            pp      = mesh.bc_to_point(bcs)
            r       = bm.linalg.norm(pp, axis=-1)
            isInPML = r > options['r0']
            F       = pml.jacobi(pp[isInPML])
            detF    = pml.detjacobi(pp[isInPML])

            NC = mesh.number_of_cells()
            NQ = bcs.shape[0]
            is_air, is_sub = options["air_region"], options["sub_region"]
            eps_air, eps_sub = options["epsilon_air"], options["epsilon_sub"]

            val = bm.zeros((NC, NQ, 3, 3), dtype=bm.complex128, device=mesh.device)
            for i in range(3):
                val = bm.set_at(val, (is_air, ALL, i, i), k**2 * eps_air)
                val = bm.set_at(val, (is_sub, ALL, i, i), k**2 * eps_sub)

            FTF    = bm.einsum("...ij, ...ik -> ...jk", F, F)
            invFTF = bm.linalg.inv(FTF)
            val[isInPML] = - k**2 * eps_air * detF[:, None, None] * invFTF

            return val

        @cartesian
        def source(p):
            shape = p.shape[:-1]
            return bm.zeros((*shape, 3), dtype=bm.complex128)

        @barycentric
        def dirichlet(bcs, index):
            shape = bcs.shape[:-1] + (3,)
            val = bm.zeros(shape, dtype=bcs.dtype, device=bcs.device)
            return val

        bform = BilinearForm(space)
        DI = CurlCurlIntegrator(diffusion, q=p+2)
        DM = ScalarMassIntegrator(reaction, q=p+2)
        bform.add_integrator(DI, DM)

        lform = LinearForm(space)
        SI = ScalarSourceIntegrator(source, q=p+2)
        lform.add_integrator(SI)

        F = lform.assembly()
        isDFace = options["pec_face"]

        b, e2d = MSA.edge_basis_integral(space, options["lumped_edge"])
        dbc = MSA.Operator(bform, dirichlet, isDFace, e2d, b)
        uh = dbc.init_solution()
        F = dbc.apply(F, uh, 1.0)

        return dbc, F, uh

    @staticmethod
    def interpolate(space, gd, uh, face_index=None):
        from fealpy.backend import bm

        p = space.p
        mesh = space.mesh
        gdof = space.number_of_global_dofs()
        isDDof = bm.zeros(gdof, device=space.device, dtype=bm.bool)
        index1 = face_index

        if p > 0:
            qf = mesh.quadrature_formula(p+2, "face")
            bcs, ws = qf.get_quadrature_points_and_weights()
            fbasis = space.face_basis(bcs)[index1] # (NF, NQ, ldof, GD)
            fm = mesh.entity_measure('face')[index1]
            M = bm.einsum("cqlg, cqmg, q, c -> clm", fbasis, fbasis, ws, fm)
            Minv = bm.linalg.inv(M)
            del M

            n = mesh.face_unit_normal()[index1, None, :]
            h2 = gd(bcs, index1)
            print(n.shape, h2.shape)
            h2 = bm.cross(n, h2)
            F = bm.einsum("cqld, cqd, q, c -> cl", fbasis, h2, ws, fm)

            face2dof = space.dof.face_to_dof()[index1]
            uh[face2dof] = bm.einsum("cl, clm -> cm", F, Minv)
            del F
            isDDof[face2dof] = True

        NE = mesh.number_of_edges()
        f2e = mesh.face_to_edge()[index1]
        bdeflag = bm.zeros(NE, device=space.device, dtype=bm.bool)
        bdeflag[f2e] = True
        index2 = bm.nonzero(bdeflag)[0]

        qf = mesh.quadrature_formula(p+2, "edge")
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = space.bspace.basis(bcs, p=p)
        em = mesh.entity_measure('edge')[index2]
        M = bm.einsum("eql, eqm, q, e -> elm", bphi, bphi, ws, em)
        Minv = bm.linalg.inv(M)
        del M

        n1 = mesh.face_unit_normal() # (NF, 3)
        n2 = bm.zeros((NE, 3), device=space.device, dtype=mesh.itype)
        count2 = bm.zeros(NE, device=space.device, dtype=mesh.itype)
        for i in range(3):
            n2 = bm.index_add(n2, f2e[:, i], n1[index1], axis=0)
            count2 = bm.index_add(count2, f2e[:, i], bm.ones_like(index1))
        n2 = n2[index2]/count2[index2, None]
        # n2[mesh.face_to_edge(), :] = n1[index2, None, :] # TODO: check this
        n2 = n2[:, None, :]

        h1 = gd(bcs, index2)
        h1 = bm.cross(n2, h1)
        t = mesh.edge_tangent()[index2]/em[:, None]
        b = bm.einsum('eqd, ed -> eq', h1, t)
        F = bm.einsum('eql, eq, q, e -> el', bphi, b, ws, em)

        edge2dof = space.dof.edge_to_dof()[index2]
        uh[edge2dof] = bm.einsum('el, elm -> em', F, Minv)
        del F
        isDDof[edge2dof] = True

        return uh, isDDof

    @staticmethod
    def edge_basis_integral(space, edge_index):
        from fealpy.backend import bm
        mesh = space.mesh
        p = space.p
        qf = mesh.quadrature_formula(p+2, "edge")
        bcs, ws = qf.get_quadrature_points_and_weights()
        bphi = space.bspace.basis(bcs, p=p)
        em = mesh.entity_measure('edge')[edge_index]
        b = bm.einsum("q, eql, e -> el", ws, bphi, em) # (NE, ldof)
        edge2dof = space.dof.edge_to_dof()[edge_index] # (NE, ldof)
        return b, edge2dof

    class Operator:
        def __init__(self, form, gd, isDFace, edge2dof, edge_basis_int):
            self.form = form
            self.gd = gd
            self.is_dirichlet_face = isDFace
            self.edge_basis_int = edge_basis_int
            self.edge2dof = edge2dof
            self.shape = form.shape

        def init_solution(self):
            from fealpy.backend import bm
            uh = bm.zeros(self.shape[1]+1, dtype=bm.complex128)
            _, self.is_dirichlet_dof = MicrostripAntenna3D.interpolate(
                self.form._spaces[0],
                self.gd, uh[:-1], self.is_dirichlet_face
            )
            return uh

        def apply(self, F, uh, integral: float):
            from fealpy.backend import bm
            uh = uh[:-1]
            F = F - self.form @ uh
            F = bm.set_at(F, self.is_dirichlet_dof, uh[self.is_dirichlet_dof])
            extra = bm.full((1,), integral, dtype=F.dtype, device=F.device)
            return bm.concat([F, extra])

        def __matmul__(self, u):
            from fealpy.backend import bm
            edge2dof = self.edge2dof

            v = bm.copy(u)
            val = v[:-1][self.is_dirichlet_dof]
            bm.set_at(v[:-1], self.is_dirichlet_dof, 0.0)
            v[:-1] = self.form @ v[:-1]
            bm.set_at(v[:-1], self.is_dirichlet_dof, val)

            extra1 = v[-1] * self.edge_basis_int
            print(f"[:-1] -> {extra1}")
            extra2 = bm.einsum('el,el->', self.edge_basis_int, v[edge2dof])
            print(f"[-1] -> {extra2}")
            v = bm.index_add(v, edge2dof, extra1)
            v[-1] += extra2

            return v
