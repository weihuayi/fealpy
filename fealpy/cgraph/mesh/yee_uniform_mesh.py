from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["YeeUniformMesh"]


class YeeUniformMesh(CNodeType):
    r"""Yee uniform mesh with material properties, sources and objects.
    
    Inputs:
        domain: Computational domain [xmin, xmax, ymin, ymax] or 3D.
        n: Number of grid points in each direction.
        mp: Material properties (eps, mu).
        source_configs: source configurations.
        object_configs: object configurations.
    
    Outputs:
        mesh: YeeUniformMesher instance with attached properties.
    """
    TITLE: str = "Yee均匀网格"
    PATH : str = "examples.CEM"
    INPUT_SLOTS = [
        PortConf("domain", DataType.TEXT, 0, 
                desc="计算域边界，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]",
                title="计算域", default=[0, 5e-6, 0, 5e-6]),
        PortConf("n", DataType.INT, 0, 
                desc="每个方向的网格剖分数",
                title="剖分数", default=50),
        PortConf("mp", DataType.DICT, 1, 
                desc="材料属性字典, 包含eps(介电常数)和mu(磁导率)",
                title="材料属性", default={"eps": 1.0, "mu": 1.0}),
        PortConf("source_configs", DataType.DICT, 1, desc="点源配置", title="点源配置"),
        PortConf("object_configs", DataType.DICT, 1, desc="物体配置", title="物体配置")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="Yee网格")
    ]
    
    @staticmethod
    def run(**options):
        import math
        from fealpy.backend import bm
        from fealpy.cem.mesh.yee_uniform_mesher import YeeUniformMesher
        
        value = options.get("domain")
        n = options.get("n", 50)
        mp = options.get("mp")
        source_configs = options.get("source_configs")
        object_configs = options.get("object_configs")
        
        if isinstance(value, str):
            domain = bm.tensor(eval(value, None, vars(math)), dtype=bm.float64)
        else:
            domain = bm.tensor(value, dtype=bm.float64)
        
        # 创建Yee网格
        if len(domain) == 4:  # 2D: [xmin, xmax, ymin, ymax]
            mesh = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=0)
        else:  # 3D: [xmin, xmax, ymin, ymax, zmin, zmax]
            mesh = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=n)
        
        # ===== 附加材料属性 =====    
        if mp is not None and len(mp) > 0:
            if isinstance(mp, list):
                mp_dict = mp[0] if len(mp) > 0 else {"eps": 1.0, "mu": 1.0}
            elif isinstance(mp, dict):
                mp_dict = mp
            else:
                mp_dict = {"eps": 1.0, "mu": 1.0}
        
            mesh.eps = mp_dict.get("eps", 1.0)
            mesh.mu = mp_dict.get("mu", 1.0)
        
        # ===== 附加源配置 =====
        if source_configs is not None and len(source_configs) > 0:
            mesh.source_configs = source_configs
        else:
            mesh.source_configs = []
            
        # ===== 附加物体配置 =====
        if object_configs is not None and len(object_configs) > 0:
            mesh.object_configs = object_configs
            
            # 创建材料场数组
            if len(domain) == 4:  # 2D
                # 初始化为背景材料
                eps_field = bm.ones((n+1, n+1)) * mesh.eps
                mu_field = bm.ones((n+1, n+1)) * mesh.mu

                # 计算网格步长
                dx = float((domain[1] - domain[0]) / n)
                dy = float((domain[3] - domain[2]) / n)
                
                # 为每个物体设置材料属性
                for obj in object_configs:
                    box = obj.get('box')
                    obj_eps = obj.get('eps')
                    obj_mu = obj.get('mu')
                    
                    if box is not None and len(box) >= 4:
                        # 计算物体在网格中的索引范围
                        i_start = int((box[0] - float(domain[0])) / dx)
                        i_end = int((box[1] - float(domain[0])) / dx)
                        j_start = int((box[2] - float(domain[2])) / dy)
                        j_end = int((box[3] - float(domain[2])) / dy)
                        
                        # 边界检查
                        i_start = max(0, min(i_start, n))
                        i_end = max(0, min(i_end, n+1))
                        j_start = max(0, min(j_start, n))
                        j_end = max(0, min(j_end, n+1))
                        
                        # 设置物体区域的材料属性
                        if obj_eps is not None:
                            eps_field[j_start:j_end, i_start:i_end] = obj_eps
                        if obj_mu is not None:
                            mu_field[j_start:j_end, i_start:i_end] = obj_mu
                
                mesh.eps_field = bm.tensor(eps_field, dtype=bm.float64)
                mesh.mu_field = bm.tensor(mu_field, dtype=bm.float64)
                
            else:  # 3D
                eps_field = bm.ones((n+1, n+1, n+1)) * mesh.eps
                mu_field = bm.ones((n+1, n+1, n+1)) * mesh.mu

                dx = float((domain[1] - domain[0]) / n)
                dy = float((domain[3] - domain[2]) / n)
                dz = float((domain[5] - domain[4]) / n)
                
                for obj in object_configs:
                    box = obj.get('box')
                    obj_eps = obj.get('eps')
                    obj_mu = obj.get('mu')
                    
                    if box is not None and len(box) == 6:
                        i_start = int((box[0] - float(domain[0])) / dx)
                        i_end = int((box[1] - float(domain[0])) / dx)
                        j_start = int((box[2] - float(domain[2])) / dy)
                        j_end = int((box[3] - float(domain[2])) / dy)
                        k_start = int((box[4] - float(domain[4])) / dz)
                        k_end = int((box[5] - float(domain[4])) / dz)
                        
                        i_start = max(0, min(i_start, n))
                        i_end = max(0, min(i_end, n+1))
                        j_start = max(0, min(j_start, n))
                        j_end = max(0, min(j_end, n+1))
                        k_start = max(0, min(k_start, n))
                        k_end = max(0, min(k_end, n+1))
                        
                        if obj_eps is not None:
                            eps_field[k_start:k_end, j_start:j_end, i_start:i_end] = obj_eps
                        if obj_mu is not None:
                            mu_field[k_start:k_end, j_start:j_end, i_start:i_end] = obj_mu
                
                mesh.eps_field = bm.tensor(eps_field, dtype=bm.float64)
                mesh.mu_field = bm.tensor(mu_field, dtype=bm.float64)
        else:
            # 没有物体，创建均匀材料场
            mesh.object_configs = []
            if len(domain) == 4:
                mesh.eps_field = bm.ones((n+1, n+1), dtype=bm.float64) * mesh.eps
                mesh.mu_field = bm.ones((n+1, n+1), dtype=bm.float64) * mesh.mu
            else:
                mesh.eps_field = bm.ones((n+1, n+1, n+1), dtype=bm.float64) * mesh.eps
                mesh.mu_field = bm.ones((n+1, n+1, n+1), dtype=bm.float64) * mesh.mu
        
        mesh.metadata = {
            'domain': domain.tolist() if hasattr(domain, 'tolist') else list(domain),
            'grid_size': n,
            'dimension': 2 if len(domain) == 4 else 3,
            'num_sources': len(mesh.source_configs),
            'num_objects': len(mesh.object_configs),
            'background_eps': float(mesh.eps),
            'background_mu': float(mesh.mu),
            'has_material_variation': len(mesh.object_configs) > 0
        }
        
        return mesh