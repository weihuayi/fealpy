import numpy as np

class PressWorkIntegrator:
    def __init__(self, q=None):
        self.q = q 

    def assembly_cell_matrix(self, trialspace, testspace, index=np.s_[:], cellmeasure=None, out=None):
        """
        construct the (pI, nabla v) fem matrix
        """ 
        mesh = trialspace[0].mesh
        
        trial_space = trialspace[0]
        test_space = testspace[0]
        
        trial_D = len(trialspace)
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
        test_gradphi = test_space.grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)
        trial_phi = trial_space.basis(bcs, index=index) # (NQ, NC, ldof)

        NC = len(cellmeasure)
        if out is None:
            K = np.zeros((NC, test_ldof*test_D, trial_D*trial_ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, test_ldof*test_D, trial_D*trial_ldof)
            K = out
        if trial_space.doforder == 'sdofs': # 标量自由度优先排序 
            for i in range(test_D):
                for j in range(trial_D):
                    val = np.einsum('i, ijm, ijn, j->jnm', ws, trial_phi, test_gradphi[..., i], cellmeasure, optimize=True)
                    K[:, i*test_ldof:(i+1)*test_ldof, j*trial_ldof:(j+1)*trial_ldof] =val 
                        
        elif trial_space.doforder == 'vdims':
            for i in range(test_D):
                for j in range(trial_D):
                    val = np.einsum('i, ijm, ijn, j->jnm', ws, trial_phi, test_gradphi[..., i], cellmeasure, optimize=True)
                    K[:, i::test_D, j::trial_D] += val 
        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass
