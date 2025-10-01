from ..nodetype import CNodeType, PortConf, DataType

class UAVPathPlanning(CNodeType):
    TITLE: str = "UAV 路径规划"
    PATH: str = "优化.路径规划"
    INPUT_SLOTS = [
        PortConf("start_x", DataType.FLOAT, min_val=900, max_val=1045),
        PortConf("start_y", DataType.FLOAT, min_val=0, max_val=200),
        PortConf("end_x", DataType.FLOAT, min_val=0, max_val=400),
        PortConf("end_y", DataType.FLOAT, min_val=600, max_val=879),
        PortConf("opt_alg", DataType.MENU, default="GeneticAlg", items=[
            "GeneticAlg", 
            "ParticleSwarmOpt", 
            "CrayfishOptAlg", 
            "HoneybadgerAlg", 
            "SnowAblationOpt", 
            "RimeOptAlg",
            "AnimatedOatOpt"
        ])
    ]  
    OUTPUT_SLOTS = [
        PortConf("PATH", DataType.TENSOR),
        PortConf("Distance", DataType.FLOAT)
    ]

    @staticmethod
    def run(start_x, start_y, end_x, end_y, opt_alg):
        from fealpy import opt
        from fealpy.pathplanning.model import PathPlanningModelManager
        from fealpy.pathplanning import TerrainLoader
        from fealpy.backend import backend_manager as bm

        terrain_data = TerrainLoader.load_terrain('ChrismasTerrain.tif')

        threats = bm.array([
            [400, 500, 200, 50],   
            [600, 200, 200, 40],   
            [500, 350, 200, 50],  
            [350, 200, 200, 40],   
            [700, 550, 200, 40],      
            [650, 750, 150, 50],   
            [300, 400, 180, 45],     
            [750, 300, 190, 55],    
            [550, 600, 170, 35],     
            [200, 500, 160, 60],     
            [450, 100, 200, 40],     
            [650, 450, 185, 50]      
        ])

        start_pos = bm.array([start_x, start_y, 150])
        end_pos = bm.array([end_x, end_y, 150])

        OptClass = getattr(opt, opt_alg)

        options = {
            'threats': threats,
            'terrain_data': terrain_data,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'opt_method': OptClass
        }

        manager = PathPlanningModelManager('route_planning')
        text = manager.get_example(1, **options)

        sol, _ = text.solver(n=10)
        x, y, z, _ = text.spherical_to_cart(sol)

        x_all = bm.concatenate(
            [
                bm.array([[text.start_pos[0]]]),
                x,
                bm.array([[text.end_pos[0]]])
            ], axis=1
        )
        y_all = bm.concatenate(
            [
                bm.array([[text.start_pos[1]]]),
                y,
                bm.array([[text.end_pos[1]]])
            ], axis=1
        )
        z_all = bm.concatenate(
            [
                bm.array([[text.start_pos[2]]]),
                z,
                bm.array([[text.end_pos[2]]])
            ], axis=1
        )

        route = bm.stack([x_all, y_all, z_all], axis=-1).squeeze()
        diff = route[:-1] - route[1:]
        total_distance = bm.sum(bm.linalg.norm(diff, axis=1))

        return route, total_distance
