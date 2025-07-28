from fealpy.backend import backend_manager as bm
from ..model import ComputationalModel

class TwoGridModel(ComputationalModel):
    def __init__(self, coarsen_model, fine_model):
        super().__init__(pbar_log=True, log_level='INFO')
        self.coarsen_model = coarsen_model
        self.fine_model = fine_model


    def refine_and_interpolate(self, k, *functions):
        if not functions:
            return ()
        
        mesh = self.coarsen_model.mesh

        data_for_bisect = {}
        component_info = {}

        for i, func_array in enumerate(functions):
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
                
                # 假设向量函数存储格式为 [u0_dof0, u1_dof0, u0_dof1, u1_dof1, ...]
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
        
        # --- 第2步: 执行网格加密 ---
        option = mesh.bisect_options(data=data_for_bisect, disp=False)
        mesh.bisect(None, option)
        self.coarsen_model.fem.update_mesh(mesh)  # 此步骤会更新所有相关的空间定义

        # --- 第3步: 在细网格上重构函数 ---
        new_functions_list = [None] * len(functions)
        retrieved_data = option['data']

        for name_prefix, info in component_info.items():
            new_space = info['space']  # 假设 fem.update_mesh 已将其更新为细网格空间
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
                stacked_arrays = [c[:] for c in new_components]
                reconstructed_flat_array = bm.stack(stacked_arrays, axis=1).T.flatten()
                new_func[:] = reconstructed_flat_array
                new_functions_list[original_index] = new_func

        return tuple(new_functions_list)

