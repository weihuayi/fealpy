import time
from fealpy.experimental.backend import backend_manager as bm
import matplotlib.pyplot as plt
import random
from fealpy.iopt.ANT_TSP import calD
# bm.set_backend('pytorch')

# 导入数据(34个城市)
citys = bm.array([
    [101.7400, 6.5600],
    [112.5300, 37.8700],
    [121.4700, 31.2300],
    [119.3000, 26.0800],
    [106.7100, 26.5700],
    [103.7300, 36.0300],
    [111.6500, 40.8200],
    [120.1900, 30.2600],
    [121.3000, 25.0300],
    [106.5500, 29.5600],
    [106.2700, 38.4700],
    [116.4000, 39.9000],
    [118.7800, 32.0400],
    [114.1700, 22.3200],
    [104.0600, 30.6700],
    [108.9500, 34.2700],
    [117.2000, 39.0800],
    [117.2700, 31.8600],
    [113.5400, 22.1900],
    [102.7300, 25.0400],
    [113.6500, 34.7600],
    [123.3800, 41.8000],
    [114.3100, 30.5200],
    [113.2300, 23.1600],
    [91.1100, 29.9700],
    [117.0000, 36.6500],
    [125.3500, 43.8800],
    [113.0000, 28.2100],
    [110.3500, 20.0200],
    [87.6800, 43.7700],
    [114.4800, 38.0300],
    [126.6300, 45.7500],
    [115.8900, 28.6800],
    [108.3300, 22.8400]
])

# 距离矩阵
D = calD(citys)

# 初始化参数
m = 10  # 蚂蚁数量
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.5  # 信息素挥发因子
Q = 1  # 常系数
Eta = 1 / D  # 启发函数
iter_max = 100  # 最大迭代次数

n = D.shape[0]
Tau = bm.ones((n, n), dtype=bm.float64)

Table = bm.zeros((m, n), dtype=int)  # 路径记录表，每一行代表一个蚂蚁走过的路径

Route_best = bm.zeros((iter_max, n), dtype=int)  # 各代最佳路径
Length_best = bm.zeros(iter_max)  # 各代最佳路径的长度

# 设置循环次数
l = 1 

# 存储每次循环的结果
shortest_lengths = []  
shortest_routes = []  

# 循环迭代寻找最佳路径
start_time = time.time()  # 记录开始时间

for _ in range(l):
    # 迭代寻找最佳路径
    for iter in range(iter_max):
        # 随机产生各个蚂蚁的起点城市
        start = list([random.randint(0, n - 1)for _ in range(m)])
        # start = bm.random.choice(n, m)
        Table[:, 0] = bm.array(start)
        citys_index = bm.arange(n)
        
        # 逐个蚂蚁路径选择
        for i in range(m):
            # 逐个城市路径选择
            for j in range(1, n):
                tabu = Table[i, :j]  # 已访问的城市集合

                allow_index = list(set(citys_index).difference(tabu))
                allow = citys_index[allow_index]  # 待访问的城市集合
                P = bm.zeros(len(allow))

                # 计算城市间转移概率
                for k, city in enumerate(allow):
                    P[k] = (Tau[tabu[-1], city]**alpha) * (Eta[tabu[-1], city]**beta)
                P = P / bm.sum(bm.array(P))

                # 轮盘赌法选择下一个访问城市
                Pc = bm.cumsum(P , axis=0)
                target_index = bm.where(Pc >= bm.random.rand(1))[0]
                target = allow[target_index[0]]
                Table[i, j] = target

        # 计算各个蚂蚁的路径距离
        Length = bm.zeros(m)
        for i in range(m):
            Route = Table[i]
            for j in range(n - 1):
                Length[i] += D[Route[j], Route[j + 1]]
            Length[i] += D[Route[-1], Route[0]]

        # 计算最短路径距离及平均距离
        if iter == 0:
            min_index = bm.argmin(Length)
            Length_best[iter] = Length[min_index]
            Route_best[iter] = Table[min_index]
        else:
            min_index = bm.argmin(Length)
            Length_best[iter] = min(Length_best[iter - 1], Length[min_index])
            if Length_best[iter] == Length[min_index]:
                Route_best[iter] = Table[min_index]
            else:
                Route_best[iter] = Route_best[iter - 1]

        # 更新信息素
        Delta_Tau = bm.zeros((n, n))
        for j in range(n - 1):
            Delta_Tau[Table[:, j], Table[:, j + 1]] += Q / Length[:]
            Delta_Tau[Table[i, -1], Table[i, 0]] += Q / Length[i]
        Tau = (1 - rho) * Tau + Delta_Tau
        
    # 结果显示
    Shortest_Length = bm.min(Length_best, keepdims = True)
    index = bm.argmin(Length_best)
    Shortest_Route = Route_best[index]

    # 在每次迭代结束后，将最短距离和最短路径添加到列表中
    shortest_lengths.append(Shortest_Length)

    SR_list = Shortest_Route.tolist()
    result_list = SR_list + [SR_list[0]]
    shortest_routes.append(bm.array(result_list))

end_time = time.time()  # 记录结束时间
elapsed_time = (end_time - start_time) / l  # 计算运行时间

# 找到最短距离的索引
min_distance_index = bm.argmin(bm.array(shortest_lengths))

# 根据索引找到最短路径
Best_length = shortest_lengths[min_distance_index]
Best_path = shortest_routes[min_distance_index]

# 找到最短距离迭代
closest_index = bm.argmin(bm.abs(bm.array(shortest_lengths) - Best_length))

# 输出最短距离和对应路径
print(f"运行时间: {elapsed_time}秒")
print('Shortest distance:', Best_length)
print('Shortest path:', Best_path)

# 绘图
plt.figure(1)
x_values = bm.concatenate((citys[Best_path, 0], [citys[Best_path[0], 0]]))
y_values = bm.concatenate((citys[Best_path, 1], [citys[Best_path[0], 1]]))
plt.plot(x_values, y_values, 'o-')

for i in range(citys.shape[0]): 
    plt.text(citys[i, 0], citys[i, 1], f'   {i + 1}')
plt.scatter(citys[Best_path[0], 0], citys[Best_path[0], 1], color='red', s=100) 
plt.xlabel('x of city location')
plt.ylabel('y of city location')
plt.title(f'Ant(shortest distance): {Best_length}\ntime: {elapsed_time}') # 

plt.figure(2)
plt.plot(range(iter_max), Length_best, 'b')
plt.legend(['shortest distance'])
plt.xlabel('number of iterations')
plt.ylabel('distance')
plt.title('changes of iterations')

plt.show()