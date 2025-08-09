from fealpy.backend import backend_manager as bm
from ..model import ComputationalModel

class TwoGridModel(ComputationalModel):
    def __init__(self, coarsen_model, fine_model):
        super().__init__(pbar_log=True, log_level='INFO')
        self.coarsen_model = coarsen_model
        self.fine_model = fine_model


    def refine_and_interpolate(self, k=1, *functions):
        if not functions:
            return ()
        assert k >= 1, "k must be at least 1"  
        current_functions = functions
        for _ in range(k):
            # --- 在每次循环中，都执行一次完整的“准备-加密-重构”流程 ---

            # mesh 对象在循环中会被 bisect 修改，所以每次都从当前函数空间获取
            mesh = current_functions[0].space.mesh

            data_for_bisect = {}
            component_info = {}

            for i, func_array in enumerate(current_functions):
                # 检查网格一致性
                if func_array.space.mesh is not mesh:
                    raise ValueError("所有输入的函数必须定义在同一个网格上。")
                
                name_prefix = f"func_{i}"
                space = func_array.space
                is_vector = hasattr(space, 'scalar_space') 
            
                if not is_vector:  # 处理标量函数
                    c2d = func_array.space.cell_to_dof()
                    data_for_bisect[name_prefix] = func_array[c2d]
                    component_info[name_prefix] = {
                        'type': 'scalar', 
                        'original_index': i, 
                        'space': func_array.space
                    }
                else:  # 处理向量函数
                    num_components = space.shape[0]
                    scalar_space = space.scalar_space
                    c2d = scalar_space.cell_to_dof()
                    
                    decomposed_array = func_array.reshape(num_components, -1).T
                    
                    comp_names = []
                    for j in range(num_components):
                        comp_name = f"{name_prefix}_comp{j}"
                        comp_array = decomposed_array[:, j]
                        data_for_bisect[comp_name] = comp_array[c2d]
                        comp_names.append(comp_name)
                    
                    component_info[name_prefix] = {
                        'type': 'vector',
                        'original_index': i,
                        'space': space,
                        'comp_names': comp_names
                    }
            
            # --- 第2步: 执行单次网格加密 ---
            option = mesh.bisect_options(data=data_for_bisect, disp=False)
            mesh.bisect(None, option)
            # self.coarsen_model.fem.update_mesh(mesh) # 原代码中这行在self.coarsen_model下
            self.coarsen_model.fem.update_mesh(mesh) 

            # --- 第3步: 在当前细网格上重构函数 ---
            new_functions_list = [None] * len(current_functions)
            retrieved_data = option['data']

            for name_prefix, info in component_info.items():
                new_space = info['space'] 
                original_index = info['original_index']

                if info['type'] == 'scalar':
                    new_func = new_space.function()
                    new_c2d = new_space.cell_to_dof()
                    new_func[new_c2d.reshape(-1)] = retrieved_data[name_prefix].reshape(-1)
                    new_functions_list[original_index] = new_func
                
                elif info['type'] == 'vector':
                    new_scalar_space = new_space.scalar_space
                    new_c2d = new_scalar_space.cell_to_dof()
                    
                    new_components = []
                    for comp_name in info['comp_names']:
                        comp_func = new_scalar_space.function()
                        comp_func[new_c2d.reshape(-1)] = retrieved_data[comp_name].reshape(-1)
                        new_components.append(comp_func)
                    
                    new_func = new_space.function()
                    # 假设 bm 是 numpy 或其兼容库
                    stacked_arrays = [c[:] for c in new_components]
                    reconstructed_flat_array = bm.stack(stacked_arrays, axis=1).T.flatten()
                    new_func[:] = reconstructed_flat_array
                    new_functions_list[original_index] = new_func

            # 关键改动：将本次循环产生的新函数列表作为下一次循环的输入
            current_functions = tuple(new_functions_list) 
        return current_functions       


    def correct_equation(self, u_refine, u_star):
        from fealpy.decorator import barycentric
        from fealpy.fem import LinearForm, LinearBlockForm, SourceIntegrator
        
        fine_model = self.fine_model
        
        BForm, _ = fine_model.linear_system()
        fine_model.update(u_refine)
        A = BForm.assembly()
        
        L0 = LinearForm(fine_model.fem.uspace)
        u_LSI = SourceIntegrator(q=fine_model.fem.q)
        u_source_LSI = SourceIntegrator(q=fine_model.fem.q)
        L0.add_integrator(u_LSI) 
        L0.add_integrator(u_source_LSI)
        L1 = LinearForm(fine_model.fem.pspace)
        LForm = LinearBlockForm([L0, L1])

        cc = fine_model.equation.coef_convection 
        cbf = fine_model.equation.coef_body_force 

        @barycentric
        def source_coef(bcs,index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result = cccoef*bm.einsum('...j, ...ij -> ...i', 
                                      u_refine(bcs, index), u_star.grad_value(bcs, index))
            result += cccoef*bm.einsum('...j, ...ij -> ...i', 
                                      u_star(bcs, index), 
                                      u_refine.grad_value(bcs, index) - u_star.grad_value(bcs, index))
            return result
        
        u_LSI.source = source_coef 
        u_source_LSI.source = cbf 
        b = LForm.assembly()

        A, b = fine_model.fem.apply_bc(A, b, fine_model.pde)
        A, b = fine_model.fem.lagrange_multiplier(A, b)
        x = fine_model.solve(A, b)

        ugdof = fine_model.fem.uspace.number_of_global_dofs()
        u = fine_model.fem.uspace.function()
        p = fine_model.fem.pspace.function()
        u[:] = x[:ugdof]
        p[:] = x[ugdof:-1] 
        return u, p



