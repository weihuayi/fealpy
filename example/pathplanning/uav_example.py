from fealpy.pathplanning import TerrainLoader
from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.pathplanning.model import PathPlanningModelManager

# 1. 读取地图
terrain_data = TerrainLoader.load_terrain('ChrismasTerrain.tif')

# 2. 构建威胁区
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

# 3. 起点、终点
start_pos = bm.array([800, 100, 150])
end_pos = bm.array([100, 800, 150])

# 4. 构建模型选项
options = {
    'threats': threats,
    'terrain_data': terrain_data,
    'start_pos': start_pos,
    'end_pos': end_pos,
    'opt_method': SnowAblationOpt
}

# 5. 初始化模型
manager = PathPlanningModelManager('route_planning')
text = manager.get_example(1, **options)

# 6. 生成路径
sol, f = text.solver(n=10)

# 7. 可视化
text.visualization(sol)
