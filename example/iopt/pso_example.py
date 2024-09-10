from fealpy.iopt.particle_swarm_opt_alg import  PSOProblem, PSO, QPSO
from fealpy.experimental.backend import backend_manager as bm
import time

# bm.set_backend('pytorch')

#例：20*20虚拟地图
#坐标：
#[[0,0]  [1,0] ………… [19,0]]
#[[0,1]  [1,1] ………… [19,1]]
#[ …………   …………       ………… ]
#[[0,19] [1,19]…………[19,19]]

start_time = time.perf_counter()

MAP=bm.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
              [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
              [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#起点终点坐标
dataS=[0, 0]
dataE=[19, 19]

if MAP[dataS[0]][dataS[1]] != 0 or MAP[dataE[0]][dataE[1]] != 0: 
    print("Error: Wrong start point or end point") # 判断起点终点坐标是否有效
else:
    textMAP = PSOProblem(MAP,dataS,dataE)
    textMAP.builddata() # 建立地图字典
    fobj = lambda x: textMAP.fitness(x)
    
    # 算法参数
    N = 20
    MaxIT = 50
    lb = 0
    ub = 1
    dim = textMAP.data["landmark"].shape[0]
    
    # 调用算法
    test1 = QPSO(N, dim, ub, lb, MaxIT, fobj)
    test1.cal()
    
    result = textMAP.calresult(test1.gbest)
    
    # 输出
    result["path"] = [x for x, y in zip(result["path"], result["path"][1:] + [None]) if x != y]
    print('The optimal path distance: ', test1.gbest_f)
    print("The optimal path: ", result["path"])
    end_time = time.perf_counter()
    running_time = end_time - start_time
    textMAP.printMAP(result, running_time)

    