from fealpy.pathplanning import UAVPathPlanning, TerrainLoader
from fealpy.backend import backend_manager as bm
from fealpy.opt import *

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

# 起点、终点
start_pos = bm.array([300, 300, 150])
end_pos = bm.array([800, 500, 150])

model = UAVPathPlanning(threats, terrain_data, start_pos, end_pos, AnimatedOatOpt)
sol, f = model.opt(n=10)
# model.optimizer.print_optimal_result()
model.output_solution(sol)
