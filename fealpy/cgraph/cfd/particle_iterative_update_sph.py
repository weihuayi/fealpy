from ..nodetype import CNodeType, PortConf, DataType

class ParticleIterativeUpdateSPH(CNodeType):
    r"""Dam Break Particle Iteration Update.

    This node implements a numerical solver for dam break problems using 
    Smoothed Particle Hydrodynamics (SPH). It performs complete SPH algorithm 
    workflow including neighbor search, density update, pressure calculation, 
    velocity update, and position update over specified simulation steps.

    Inputs:
        maxstep (int): Maximum number of iteration steps
        dx (float): Horizontal particle spacing
        dy (float): Vertical particle spacing  
        rhomin (float): Reference minimum density
        dt (float): Time step size
        c0 (float): Initial sound speed
        gamma (float): Specific heat ratio
        alpha (float): Artificial viscosity coefficient
        rho0 (float): Initial particle density
        pp (tensor): Fluid particle coordinates
        bpp (tensor): Boundary particle coordinates

    Outputs:
        velocity (tensor): Fluid particle velocities
        pressure (tensor): Boundary particle pressures
    """

    TITLE: str = "溃坝问题粒子迭代更新"
    PATH: str = "流体.粒子迭代更新"
    DESC: str = """该节点实现了基于光滑粒子流体动力学(SPH)的溃坝问题数值模拟求解器，
    包含完整的SPH算法流程：邻居搜索、密度更新、压力计算、速度更新、位置更新等。"""
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, ttype=0, title="最大迭代步数"),
        PortConf("dx", dtype=DataType.FLOAT, ttype=1, title="水平粒子间隔"),
        PortConf("dy", dtype=DataType.FLOAT, ttype=1, title="垂直粒子间隔"),
        PortConf("rhomin", dtype=DataType.FLOAT, ttype=0, title="最小参考密度"),
        PortConf("dt", dtype=DataType.FLOAT, ttype=0, title="时间步长"),
        PortConf("c0", dtype=DataType.FLOAT, ttype=0, title="初始声速"),
        PortConf("gamma", dtype=DataType.FLOAT, ttype=0, title="比热容比"),
        PortConf("alpha", dtype=DataType.FLOAT, ttype=0, title="人工粘性系数"),
        PortConf("rho0", dtype=DataType.FLOAT, ttype=0, title="初始粒子密度"),
        PortConf("pp", dtype=DataType.TENSOR, ttype=1, title="流体粒子坐标"),
        PortConf("bpp", dtype=DataType.TENSOR, ttype=1, title="边界粒子坐标"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity", dtype=DataType.TENSOR, title="流体粒子速度"),
        PortConf("pressure", dtype=DataType.TENSOR, title="边界粒子压力"),
    ]
    

    @staticmethod
    def run(maxstep, dx, dy, rhomin, dt, c0, gamma, alpha, rho0, pp, bpp,output_dir):
        from fealpy.cfd.simulation.sph.particle_solver_new import BamBreakSolver,ParticleSystem
        import json
        import os
        import gzip
        from fealpy.cfd.simulation.utils import VTKWriter
        writer = VTKWriter()
        
        particles = ParticleSystem.initialize_particles(pp, bpp, rho0)
        sph_solver = BamBreakSolver(particles)
        data = []
        j = 0
        for i in range(maxstep):
            print(f"Step: {i}")
            # 邻居搜索
            idx = sph_solver.find_neighbors_within_distance(
                particles.particles["position"], 2 * sph_solver.H
            )

            # 更新步骤
            sph_solver.change_rho(idx)
            sph_solver.change_p(i)
            sph_solver.change_v(idx)
            sph_solver.change_position(idx)

            # 周期性密度重初始化

            if i % 30 == 0 and i != 0:
                sph_solver.rein_rho(idx)
                
            # current_data = {
            # "time": round(i * dt, 8),
            # "position": sph_solver.ps.particles["position"].tolist(),  # ndarray -> list
            # "velocity": sph_solver.ps.particles["velocity"].tolist(),  # ndarray -> list
            # "pressure": sph_solver.ps.particles["pressure"].tolist(),  # ndarray -> list
            # }    
            
            # path = "./dambreak/"
            # zfname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'    
            # writer.write_vtk(current_data, zfname) 
                
            os.makedirs(output_dir, exist_ok=True)
            if i % 10 == 0:
                data.append ({
                    "time": round(i * dt, 8),
                    "值":{
                        "uh" : sph_solver.ps.particles["velocity"].tolist(),  # ndarray -> list
                        "ph" : sph_solver.ps.particles["pressure"].tolist(),  # ndarray -> list
                    }, 
                    "几何": {
                        "position": sph_solver.ps.particles["position"].tolist(),  # ndarray -> list
                    }
                    })
                
                if len(data) == 10 :
                    j += 1
                    file_name = f"file_{j:08d}.json.gz"
                    file_path = os.path.join(output_dir, file_name)

                    with gzip.open(file_path, "wt", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    data.clear()
            
            
        velocity = sph_solver.ps.particles["velocity"]
        pressure = sph_solver.ps.particles["pressure"]
        return velocity, pressure
