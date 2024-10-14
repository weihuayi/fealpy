import numpy as np

class VectorDarcyIntegrator:
    """
    @note (\\nabla u, v) 其中 u 是标量, v 是向量
    """    

    def __init__(self, c = None, q=3):
        self.coef = c 
        self.q = q

    def assembly_cell_matrix(self, trialspace, testspace, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief assemble the matrix (\\nabla u, v) on each cell
        """ 
        
        trial_space = trialspace[0]
        test_space = testspace[0]

        mesh = trial_space.mesh
        
        test_D = len(testspace)
    
        trial_ldof = trial_space.number_of_local_dofs()
        test_ldof = test_space.number_of_local_dofs()
        trial_gdof = trial_space.number_of_global_dofs()
        test_gdof = test_space.number_of_global_dofs()
        trial_cell2dof = trial_space.cell_to_dof() 
        test_cell2dof = test_space.cell_to_dof() 
        
        p = trial_space.p
        q = self.q if self.q is not None else p+1

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf =  mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        trial_gphi = trial_space.grad_basis(bcs, index=index) # (NQ, NC, ldof0, GD) 
        test_phi = test_space.basis(bcs, index=index) # (NQ, NC, ldof1)

        NC = len(cellmeasure)
        if out is None:
            K = np.zeros((NC, test_ldof*test_D, trial_ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, test_ldof*test_D, trial_ldof)
            K = out
        if test_space.doforder == 'sdofs': # 标量自由度优先排序 
            for j in range(test_D):
                val = np.einsum('q, qci, qcj, c->cij', ws, test_phi, trial_gphi[..., j], cellmeasure, optimize=True)
                K[:, j*test_ldof:(j+1)*test_ldof] = val 
                        
        elif test_space.doforder == 'vdims':
            for j in range(test_D):
                val = np.einsum('q, qci, qcj, c->cij', ws, test_phi, trial_gphi[..., j], cellmeasure, optimize=True)
                K[:, j::test_D] += val 
        if out is None:
            return K

