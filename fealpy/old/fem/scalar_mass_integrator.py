import numpy as np
from fealpy.old.fem.precomp_data import data

class ScalarMassIntegrator:
    """
    @note (c u, v)
    """    

    def __init__(self, uh=None, uh_func=None, grad_uh_func=None, c=None, q=None):
        self.uh = uh
        self.uh_func = uh_func
        self.grad_uh_func = grad_uh_func
        self.coef = c
        self.q = q
        self.type = 'BL3'

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """

        q = self.q if self.q is not None else space.p+1
        coef = self.coef
        uh = self.uh
        grad_uh_func = self.grad_uh_func

        mesh = space.mesh

        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs()

        if out is None:
            M = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            M = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi0 = space.basis(bcs, index=index) # (NQ, NC, ldof)
        if coef is None:
            M += np.einsum('q, qci, qcj, c -> cij', ws, phi0, phi0, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        coef = coef(ps)
                    elif coef.coordtype == 'barycentric':
                        coef = coef(bcs, index=index)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            if np.isscalar(coef):
                M += coef*np.einsum('q, qci, qcj, c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ):
                    M += np.einsum('q, c, qci, qcj, c -> cij', ws, coef, phi0, phi0, cellmeasure, optimize=True)
                else:
                    M += np.einsum('q, qc, qci, qcj, c -> cij', ws, coef, phi0, phi0, cellmeasure, optimize=True)
            else:
                raise ValueError("coef is not correct!")

        if out is None:
            return M

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        coef = self.coef
        p = space.p
        q = self.q
        uh = self.uh
        uh_func = self.uh_func

        q = p+3 if q is None else q

        mesh = space.mesh
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.dof.number_of_local_dofs() 
        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.basis(bcs, index=index) #TODO: 考虑非重心坐标的情形

        if callable(coef):
            if hasattr(coef, 'coordtype'):
                if coef.coordtype == 'barycentric':
                    coef = coef(bcs, index=index)
                elif coef.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            else: # 默认是笛卡尔
                ps = mesh.bc_to_point(bcs, index=index)
                coef = coef(ps)
        else:
            coef = coef
        val = -coef * uh_func(uh(bcs))

        if isinstance(val, (int, float)):
            bb += val*np.einsum('q, qci, c->ci', ws, phi, cellmeasure, optimize=True)
        else:
            if val.shape == (NC, ): 
                bb += np.einsum('q, c, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
            else:
                if val.shape[-1] == 1:
                    val = val[..., 0]
                bb += np.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
        if out is None:
            return bb 

    def assembly_cell_matrix_quickly(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """

        q = self.q if self.q is not None else space.p+1
        coef = self.coef

        mesh = space.mesh

        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs()

        if out is None:
            M = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            M = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi0 = mesh.shape_function(bc=bcs, p=space.p) # (NQ, ldof)
        P = np.einsum('q, qi, qj -> ij', ws, phi0, phi0) # (ldof, ldof)

        if coef is None:
            M += np.einsum('ij, c -> cij', P, cellmeasure, optimize=True)
        else:
            if callable(coef):
                raise ValueError("coef 不支持该类型")

            if np.isscalar(coef):
                M += coef*np.einsum('ij, c -> cij', P, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray):
                raise ValueError("coef 不支持该类型")
            else:
                raise ValueError("coef is not correct!")

        if out is None:
            return M

    def assembly_cell_matrix_fast(self, space,
            trialspace=None, testspace=None, coefspace=None,
            index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 基于无数值积分的组装方式
        """
        coef = self.coef

        mesh = space.mesh 
        meshtype = mesh.type

        if trialspace is None:
            trialspace = space
            TAFtype = space.btype
            TAFdegree = space.p
            TAFldof = space.number_of_local_dofs()
        else:
            TAFtype = trialspace.btype
            TAFdegree = trialspace.p
            TAFldof = trialspace.number_of_local_dofs()  

        if testspace is None:
            testspace = trialspace
            TSFtype = TAFtype
            TSFdegree = TAFdegree
            TSFldof = TAFldof
        else:
            TSFtype = testspace.btype
            TSFdegree = testspace.p 
            TSFldof = testspace.number_of_local_dofs()

        if coefspace is None:
            coefspace = testspace
            COFtype = TSFtype
            COFdegree = TSFdegree
            COFldof = TSFldof
        else:
            COFtype = coefspace.btype
            COFdegree = coefspace.p 
            COFldof = coefspace.number_of_local_dofs()

        Itype = self.type 
        dataindex = Itype + "_" + meshtype + "_TAF_" + TAFtype + "_" + \
                str(TAFdegree) + "_TSF_" + TSFtype + "_" + str(TSFdegree)

        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                 cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)

        if out is None:
            M = np.zeros((NC, TSFldof, TAFldof), dtype=trialspace.ftype)
        else:
            M = out

        if coef is None:
            M += np.einsum('c, cij -> cij', cellmeasure, data[dataindex], optimize=True)
        else:
            if callable(coef):
                u = coefspace.interpolate(coef)
                cell2dof = coefspace.cell_to_dof()
                coef = u[cell2dof]
            if np.isscalar(coef):
                M += coef * np.einsum('c, aij -> cij', cellmeasure, data[dataindex], optimize=True)
            elif coef.shape == (NC, COFldof):
                dataindex += "_COF_" + COFtype + "_" + str(COFdegree)
                #print("data[dataindex]:", data[dataindex].shape)
                M += np.einsum('c, ijk, ck -> cij', cellmeasure, data[dataindex], coef, optimize=True)
            elif coef.shape == (NC, ):
                M += np.einsum('c, aij, c -> cij', cellmeasure, data[dataindex], coef, optimize=True)
            else:
                raise ValueError("coef is not correct!")

        if out is None:
            return M

    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
