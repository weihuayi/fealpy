from fealpy.backend import backend_manager as bm
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import random
import datetime
from random import randint
import pygame
from enum import Enum
from functools import total_ordering
from matplotlib import colors

# 定义全局变量：地图中节点的像素大小
CELL_WIDTH = 160 #单元格宽度
CELL_HEIGHT = 160 #单元格长度
BORDER_WIDTH = 10 #边框宽度
BLOCK_NUM = 50 #地图中的障碍物数量

class Color(Enum):
    ''' 颜色 '''
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    @staticmethod
    def random_color():
        '''设置随机颜色'''
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        return (r, g, b)


class Map(object):
    def __init__(self, mapsize):
        self.mapsize = mapsize

    def generate_cell(self, cell_width, cell_height):
        '''
        定义一个生成器，用来生成地图中的所有节点坐标
        :param cell_width: 节点宽度
        :param cell_height: 节点长度
        :return: 返回地图中的节点
        '''
        x_cell = -cell_width
        for num_x in range(self.mapsize[0] // cell_width):
            y_cell = -cell_height
            x_cell += cell_width
            for num_y in range(self.mapsize[1] // cell_height):
                y_cell += cell_height
                yield (x_cell, y_cell)
                #print(x_cell, y_cell)

class Node(object):
    def __init__(self, pos):
        self.pos = pos
        self.father = None
        self.gvalue = 0
        self.fvalue = 0

    def compute_fx(self, enode, father):
        if father == None:
            print('未设置当前节点的父节点！')

        gx_father = father.gvalue
        #采用欧式距离计算父节点到当前节点的距离
        gx_f2n = math.sqrt((father.pos[0] - self.pos[0])**2 + (father.pos[1] - self.pos[1])**2)
        gvalue = gx_f2n + gx_father

        hx_n2enode = math.sqrt((self.pos[0] - enode.pos[0])**2 + (self.pos[1] - enode.pos[1])**2)
        fvalue = gvalue + hx_n2enode
        return gvalue, fvalue
    
    def compute_fx2(self, enode, father):
        if father == None:
            print('未设置当前节点的父节点！')

        gx_father = father.gvalue
        #采用曼哈顿距离计算父节点到当前节点的距离
        
        gx_f2n = abs(father.pos[0] - self.pos[0]) + abs(father.pos[1] - self.pos[1])#初始状态到当前状态的代价g(n)
        
        gvalue = gx_f2n + gx_father

        hx_n2enode = abs(self.pos[0] - enode.pos[0]) + abs(self.pos[1] - enode.pos[1])#当前状态到终点状态的代价h(n)
        fvalue = gvalue + hx_n2enode   #总代价
        return gvalue, fvalue

    def set_fx(self, enode, father):
        self.gvalue, self.fvalue = self.compute_fx2(enode, father)
        self.father = father

    def update_fx(self, enode, father):
        gvalue, fvalue = self.compute_fx2(enode, father)
        if fvalue < self.fvalue:
            self.gvalue, self.fvalue = gvalue, fvalue
            self.father = father

class AStar(object):
    def __init__(self, mapsize, pos_sn, pos_en):
        self.mapsize = mapsize #表示地图的投影大小，并非屏幕上的地图像素大小
        self.openlist, self.closelist, self.blocklist = [], [], []
        self.snode = Node(pos_sn) #用于存储路径规划的起始节点
        self.enode = Node(pos_en) #用于存储路径规划的目标节点
        self.cnode = self.snode   #用于存储当前搜索到的节点

    def run(self):
        self.openlist.append(self.snode)
        while(len(self.openlist) > 0):
            #查找openlist中fx最小的节点
            fxlist = list(map(lambda x: x.fvalue, self.openlist))
            index_min = fxlist.index(min(fxlist))
            self.cnode = self.openlist[index_min]
            del self.openlist[index_min]
            self.closelist.append(self.cnode)

            # 扩展当前fx最小的节点，并进入下一次循环搜索
            self.extend(self.cnode)
            # 如果openlist列表为空，或者当前搜索节点为目标节点，则跳出循环
            if len(self.openlist) == 0 or self.cnode.pos == self.enode.pos:
                break

        if self.cnode.pos == self.enode.pos:
            self.enode.father = self.cnode.father
            return 1
        else:
            return -1

    def get_minroute(self):
        minroute = []
        current_node = self.enode

        while(True):
            minroute.append(current_node.pos)
            current_node = current_node.father
            if current_node.pos == self.snode.pos:
                break

        minroute.append(self.snode.pos)
        minroute.reverse()
        return minroute

    def extend(self, cnode):
        nodes_neighbor = self.get_neighbor(cnode)
        for node in nodes_neighbor:
            #判断节点node是否在closelist和blocklist中，因为closelist和blocklist中元素均为Node类，所以要用map函数转换为坐标集合
            if node.pos in list(map(lambda x:x.pos, self.closelist)) or node.pos in self.blocklist:
                continue
            else:
                if node.pos in list(map(lambda x:x.pos, self.openlist)):
                    node.update_fx(self.enode, cnode)
                else:
                    node.set_fx(self.enode, cnode)
                    self.openlist.append(node)

    def setBlock(self, blocklist):
        '''
        获取地图中的障碍物节点，并存入self.blocklist列表中
        注意：self.blocklist列表中存储的是障碍物坐标，不是Node类
        :param blocklist:
        :return:
        '''
        self.blocklist.extend(blocklist)
        # for pos in blocklist:
        #     block = Node(pos)
        #     self.blocklist.append(block)

    def get_neighbor(self, cnode):
        offsets = [(0,1),(-1,0),(1,0),(0,-1)]
        nodes_neighbor = []
        x, y = cnode.pos[0], cnode.pos[1]
        for os in offsets:
            x_new, y_new = x + os[0], y + os[1]
            pos_new = (x_new, y_new)
            #判断是否在地图范围内,超出范围跳过
            if x_new < 0 or x_new > self.mapsize[0] - 1 or y_new < 0 or y_new > self.mapsize[1] - 1:
                continue
            nodes_neighbor.append(Node(pos_new))

        return nodes_neighbor

    def showresult(mapsize, pos_sn, pos_en, blocklist, routelist):
        # 初始化导入的Pygame模块
        pygame.init()
        # 此处要将地图投影大小转换为像素大小，此处设地图中每个单元格的大小为CELL_WIDTH*CELL_HEIGHT像素
        mymap = Map((mapsize[0]*CELL_WIDTH, mapsize[1]*CELL_HEIGHT))
        pix_sn = (pos_sn[0]*CELL_WIDTH, pos_sn[1]*CELL_HEIGHT)
        pix_en = (pos_en[0]*CELL_WIDTH, pos_en[1]*CELL_HEIGHT)
        #对blocklist和routelist中的坐标同样要转换为像素值
        bl_pix = list(map(transform, blocklist))
        rl_pix = list(map(transform, routelist))
        # 初始化显示的窗口并设置尺寸
        screen = pygame.display.set_mode(mymap.mapsize)
        # 设置窗口标题
        pygame.display.set_caption('A*算法路径搜索演示：')
        #用白色填充屏幕
        screen.fill(Color.WHITE.value)#为什么用参数Color.WHITE不行？

        #绘制屏幕中的所有单元格
        for (x, y) in mymap.generate_cell(CELL_WIDTH, CELL_HEIGHT):
            if (x,y) in bl_pix:
                #绘制黑色的障碍物单元格，并留出2个像素的边框
                pygame.draw.rect(screen, Color.BLACK.value, ((x+BORDER_WIDTH,y+BORDER_WIDTH), (CELL_WIDTH-2*BORDER_WIDTH, CELL_HEIGHT-2*BORDER_WIDTH)))
            else:
                # 绘制绿色的可通行单元格，并留出2个像素的边框
                pygame.draw.rect(screen, Color.GREEN.value, ((x+BORDER_WIDTH,y+BORDER_WIDTH), (CELL_WIDTH-2*BORDER_WIDTH, CELL_HEIGHT-2*BORDER_WIDTH)))
        #绘制起点和终点
        pygame.draw.circle(screen, Color.BLUE.value, (pix_sn[0]+CELL_WIDTH//2, pix_sn[1]+CELL_HEIGHT//2), CELL_WIDTH//2 - 1)
        pygame.draw.circle(screen, Color.RED.value, (pix_en[0]+CELL_WIDTH//2, pix_en[1]+CELL_HEIGHT//2), CELL_WIDTH//2 - 1)

        #绘制搜索得到的最优路径
        pygame.draw.aalines(screen, Color.RED.value, False, rl_pix)
        keepGoing = True
        while keepGoing:
            pygame.time.delay(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keepGoing = False
            pygame.display.flip()



    def transform(pos):
        xnew, ynew = pos[0]*CELL_WIDTH, pos[1]*CELL_HEIGHT
        return (xnew, ynew)

class GridMap:
    
    def __init__(self, width, height, init_val=0):
        """__init__

        :param width: number of grid for width 宽
        :param height: number of grid for height 高
        :param init_val: initial value for all grid 初始值
        """
        self.width = width
        self.height = height

        self.n_data = self.width * self.height #数据点的大小
        self.data = [init_val] * self.n_data
        self.data_type = type(init_val)#数据类型
        self.obstacle = []
        
    def get_value_from_xy_index(self, x_ind, y_ind): 
        """get_value_from_xy_index
        获取对应节点的数值

        when the index is out of grid map area, return None

        :param x_ind: x index
        :param y_ind: y index
        """

        grid_ind = self.calc_grid_index_from_xy_index(x_ind, y_ind)
        if x_ind < 0 or x_ind > self.width - 1:
            return None
        if y_ind < 0 or y_ind > self.height - 1:
            return None


        if 0 <= grid_ind < self.n_data:
            return self.data[grid_ind]
        else:
            return None

    def calc_grid_index_from_xy_index(self, x_ind, y_ind):
        grid_ind = int(y_ind * self.width + x_ind)
        return grid_ind
    
    def calc_xy_index_from_grid_index(self, grid_ind):
        y_ind, x_ind = divmod(grid_ind, self.width)
        return x_ind, y_ind
    


    def set_value_from_xy_index(self, x_ind, y_ind, val):
        """set_value_from_xy_index

        return bool flag, which means setting value is succeeded or not

        :param x_ind: x index
        :param y_ind: y index
        :param val: grid value
        """

        if (x_ind is None) or (y_ind is None):
            return False, False

        grid_ind = int(y_ind * self.width + x_ind)

        if 0 <= grid_ind < self.n_data and isinstance(val, self.data_type):
            self.data[grid_ind] = val
            return True  # OK
        else:
            return False  # NG
        

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):

        val = self.get_value_from_xy_index(x_ind, y_ind)

        if val is None or val >= occupied_val:
            return True
        else:
            return False
        
    def generate_obstacle(self, p):
        self.obstacle = []
        if p >= 0 and p <= 1:
            
            obstacle_num = int(self.n_data * p)
            for i in range (obstacle_num):
                x = bm.random.randint(0, self.width)
                y = bm.random.randint(0, self.height)
                self.set_value_from_xy_index(x, y ,2.0)
                self.obstacle.append((x,y))
        else:
            print("设置的概率不在范围内")
        
    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("n_data:", self.n_data)    
        
    def plot_grid_map(self, ax=None):
        float_data_array = bm.array([d for d in self.data])
        print(float_data_array)
        grid_data = bm.reshape(float_data_array, (self.height, self.width))
        print(grid_data)
        cmap = colors.ListedColormap(['none','grey',  'black', 'red','blue', 'yellow', 'magenta', 'green', 'cyan', 'blue'])
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap=cmap, vmin=0.0, vmax=10.0)
        #fig = plt.figure(tight_layout=True,figsize = (12,12))
        plt.axis("equal")
        plt.grid(ls="--")
        plt.xticks(range(0, self.width + 1, 1),fontsize=12)  
        plt.yticks(range(0, self.height + 1, 1),fontsize=9)
        return heat_map
    
    def show_grid_data(self):
        float_data_array = bm.array([d for d in self.data])
        #print(float_data_array)
        grid_data = bm.reshape(float_data_array, (self.height, self.width))
        #print(grid_data)
        return grid_data        

class Graph:
    esp = 0.001
    def __init__(self, n, width, height):
        """
        n:    小车数量
        size: 地图大小为width * height的网格
        """
        self.n = n
        self.width = width
        self.height = height
        self.grid_map = GridMap(self.width, self.height)
        self.arrivelist = []#到达列表，固定障碍
        self.direction = bm.zeros((1,self.n))
        self.block = []#固定障碍列表
        
    def generate_block(self, m):
        """
        设置随机障碍
        input:
        m : 障碍数

        output:
        blok:障碍列表
        """
        indexs = list(range(self.width * self.height))
        if m >= 0 and m <= self.height * self.width:
            self.block = random.sample(indexs, m)
        return self.block

    def set_data_index(self):
        """
        设置起点和终点
        """
        arr_start_index = []
        arr_end_index = []
        while len(arr_start_index) < self.n:
            index = random.randint(0, self.width * self.height - 1)
            if index not in arr_start_index and index not in self.block:
                arr_start_index.append(index)
        
        while len(arr_end_index) < self.n:
            index = random.randint(0, self.width * self.height - 1)
            if index not in arr_end_index and index not in arr_start_index and index not in self.block:
                arr_end_index.append(index)     
        return bm.array(arr_start_index), bm.array(arr_end_index)    

    def sort_by_distance(self,arr_start_index, arr_end_index):
        """
        依据曼哈顿距离进行优先级排序
        """   
        distances = []
        for i in range(self.n):
            sx, sy = self.calc_xyindex(arr_start_index[i])
            ex, ey = self.calc_xyindex(arr_end_index[i])
            distance = self.manhattan_distance(sx, sy, ex, ey)
            distances.append(distance)
        sorted_start = sorted(zip(distances, list(arr_start_index)), key=lambda x: x[0])
        sorted_end = sorted(zip(distances, list(arr_end_index)), key=lambda x: x[0])
        sorted_arr_start_index = []
        sorted_arr_end_index = []
        for i in range(self.n):
            sorted_arr_start_index.append(sorted_start[i][1])
            sorted_arr_end_index.append(sorted_end[i][1])
        return bm.array(sorted_arr_start_index), bm.array(sorted_arr_end_index)     
                
    
    def calc_xyindex(self, index):
        #返回编号的x, y坐标
        y_ind, x_ind = divmod(index, self.width)
        return x_ind, y_ind

    def calc_grid_index(self, x, y):
        #返回x,y坐标对应的编号
        return int(y * self.width + x)
    
    def manhattan_distance(self, x1, y1, x2, y2):
        """
        计算曼哈顿距离
        """
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_route(self, arr_start_index, arr_end_index):
        route = []#存放全部路径
        for i in range(self.n):

            #转为二维坐标
            pos_snode = tuple(self.calc_xyindex(arr_start_index[i]))
            pos_enode = tuple(self.calc_xyindex(arr_end_index[i]))
            mapsize = (self.width, self.height)
            myAstar = AStar(mapsize, pos_snode, pos_enode)
            blocklist = []
            myAstar.setBlock(self.block)
            routelist = [] #记录搜索到的最优路径
            if myAstar.run() == 1:
                routelist = myAstar.get_minroute()
                print(routelist)
            else:
               print('路径规划失败！')
            route.append(routelist)
        return route
    
    def get_route2(self, arr_start_index, arr_end_index):
        esp = 0.001
        route = []
        blocklist = []
        mapsize = (self.width, self.height)
        for i in range(self.n):
            #转为二维坐标
            pos_snode = tuple(self.calc_xyindex(arr_start_index[i]))
            pos_enode = tuple(self.calc_xyindex(arr_end_index[i]))
            routelist = [pos_snode]
            pos_cnode = (pos_snode[0] , pos_snode[1])
            while True:
                if pos_enode[0] - pos_cnode[0] > esp:
                    pos_cnode = (pos_cnode[0] + 1, pos_cnode[1])
                    routelist.append(pos_cnode)
                elif pos_cnode[0] - pos_enode[0] > esp:
                    pos_cnode = (pos_cnode[0] - 1, pos_cnode[1])
                    routelist.append(pos_cnode)
                elif pos_enode[1] - pos_cnode[1] > esp:
                    pos_cnode = (pos_cnode[0] , pos_cnode[1] + 1)
                    routelist.append(pos_cnode)
                elif pos_cnode[1] - pos_enode[1] > esp:
                    pos_cnode = (pos_cnode[0] , pos_cnode[1] - 1)
                    routelist.append(pos_cnode)
                if abs(pos_cnode[0] - pos_enode[0]) < esp and abs(pos_cnode[1] - pos_enode[1]) < esp:
                    break
            route.append(routelist)
        return route
        
    def check_collision_index(self, result_path, t):

        #传入某个时间点的不同点所在坐标数组
        esp = 0.0001    
        collision_message = bm.zeros((1,6))
        paths_index = self.cal_result_path_index(result_path, t + 2)
     
        for t in range(0,t):
            result = paths_index[:,t]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    #冲突点在节点的情况
                    if abs(result[i] - result[j]) < esp :
                        print("时间为",t,"时",i, "与", j, "相撞","坐标为",self.calc_xyindex(result[j]))
                        collision = bm.array([int(t), int(i), int(j), result[j], 0, 0])
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[j])[0],self.calc_xyindex(result[j])[1], 7.0)
                    
                        #判断碰撞类型
                        a_previous = paths_index[int(i), t-1]
                        a_next = paths_index[int(i), t+1]
    
                        b_previous = paths_index[int(j), t-1]
                        b_next = paths_index[int(j),  t+1]
                
                        if abs(a_previous - b_next) < esp and abs(b_previous - a_next) < esp :
                    
                            print("相向冲突:", i, j)
                            collision[-1] = 1
       
                        else:
                            print("节点冲突:", i, j)
                
                        collision = collision.reshape(1, -1)
                        collision_message = bm.concatenate((collision_message, collision), axis=0)
                        
                      #冲突点不在节点的情况
                    if abs(result[i] - paths_index[j][t + 1]) < esp and abs(result[j] - paths_index[i][t + 1]) < esp:
                        print("时间为",t,"时",i, "与", j, "相撞","坐标为",self.calc_xyindex(result[i]),self.calc_xyindex(result[j]) )
                        print("相向冲突:",i,j)
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[j])[0],self.calc_xyindex(result[j])[1], 7.0)
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[i])[0],self.calc_xyindex(result[i])[1], 7.0)
                        collision = bm.array([int(t), int(i), int(j), result[i], result[j],0])
                        collision = collision.reshape(1, -1)
                        collision_message = bm.concatenate((collision_message, collision), axis=0)    
        #self.grid_map.plot_grid_map()
        return collision_message[1:]
    
    def turn_num(self, result_path, t):
        """
        计算不同点拐弯次数
        """
        esp = 0.001
        turn = bm.zeros((1,self.n))
        paths_index = self.cal_result_path_index(result_path, t + 2)
        for i in range(self.n):
            for j in range(1, t):
                x_previous, y_previous = self.calc_xyindex(paths_index[i][j - 1].tolist())
                x_next, y_next = self.calc_xyindex(paths_index[i][j + 1].tolist())
                if x_previous != x_next and y_previous != y_next :
                    turn[:,i] = turn[:,i] + 1
        return turn
    
    def cal_result_path_index(self,result_path, t):
        result_path_index = bm.zeros((self.n, t ))
        for i in range(self.n):
            for j in range(t ):
                ind = self.calc_grid_index(result_path[i][j][0], result_path[i][j][1])
                result_path_index[i][j] = ind
        #print("路径编号",result_path_index)
        return  result_path_index

    def paths_avoid_collisions(self, arr_start_index, arr_end_index, k):
        """
        考虑碰撞情况
        """
        k = 2
        t = 1
        slist = []
        elist = []
        blocklist = []
        fixed_block = []
        flat = 1
        for i in range(self.n):
            print("$$$$$$$$$$$$$$$$$$$$$$$$", i)
            #转为二维坐标
            pos_snode = tuple(self.calc_xyindex(arr_start_index[i]))
            pos_enode = tuple(self.calc_xyindex(arr_end_index[i]))
            slist.append(pos_snode)
            elist.append(pos_enode)

        path = self.get_route(arr_start_index,arr_end_index)#获取初始路径
        print("初始路径",path)
        result_path = []
        for i in range(self.n):
            pathj = [slist[i]]
            result_path.append(pathj)
            path[i].pop(0)
        print("初始路径",path)
        while True:
           
            grid_map_collisions = GridMap(self.width, self.height)#临时地图存储下1步状态
            for block in self.block:
                x , y= self.calc_xyindex(block)
                fixed_block.append((x,y))
                grid_map_collisions.set_value_from_xy_index(x, y, 2)

            replan = []#存储需要重新规划的点
            replan2 = []

            for a in self.arrivelist:
                grid_map_collisions.set_value_from_xy_index(elist[a][0], elist[a][1], 2)
                 
            for j in range(self.n):
                if j in self.arrivelist:
                    result_path[j].append(elist[j])
                    continue

                #print(path[j][0])
                value = grid_map_collisions.get_value_from_xy_index(path[j][0][0],path[j][0][1])
                #print("值",value)
                if value == 0:#按预定路线走
                    if path[j][0][0] == elist[j][0] and path[j][0][1] == elist[j][1]:#到达终点
                        grid_map_collisions.set_value_from_xy_index(path[j][0][0],path[j][0][1], 2)
                        self.arrivelist.append(j)
                        blocklist.append(elist[j])
                    else:
                        grid_map_collisions.set_value_from_xy_index(path[j][0][0],path[j][0][1], 1)

                    #避免交换位置的相向冲突
                    for k in range(j + 1, self.n):
                        if len(path[j]) > 1 and len(path[k]) > 1:
                            if path[j][0] == path[k][1] and path[j][1] == path[k][0]:
                                replan.append(k)
                                grid_map_collisions.set_value_from_xy_index(path[j][1][0],path[j][1][1], 1)
                        if len(path[j]) >= 1 and len(result_path[k]) >= 1:
                            if path[j][0] == result_path[k][-1] and path[k][0] == result_path[j][-1]:
                                replan2.append(k)
                                grid_map_collisions.set_value_from_xy_index(result_path[j][-1][0],result_path[j][-1][1], 1)

                    #记录路径
                    result_path[j].append(path[j][0])
                    #print("走的是",path[j][0])
                    path[j].pop(0)

                elif value == 1:#暂时被占用，等待或重新规划
                    if len(result_path[j]) > 0 and grid_map_collisions.get_value_from_xy_index(result_path[j][-1][0],result_path[j][-1][1]) == 0 and j not in replan2 :#可以等待
                        print("等待")
                        grid_map_collisions.set_value_from_xy_index(result_path[j][-1][0],result_path[j][-1][1], 1)
                        result_path[j].append(result_path[j][-1])
                        #避免交换位置的相向冲突
                        for k in range(j + 1, self.n):
                            if len(path[j]) > 1 and len(path[k]) > 1:
                                if path[j][0] == path[k][1] and path[j][1] == path[k][0]:
                                    replan.append(k)
                                    grid_map_collisions.set_value_from_xy_index(path[j][1][0],path[j][1][1], 1)
                            if len(path[j]) > 1 and len(result_path[k]) > 1:
                                if path[j][0] == result_path[k][-1] and path[k][0] == result_path[j][-2]:
                                    replan2.append(k)
                                    grid_map_collisions.set_value_from_xy_index(result_path[j][-2][0],result_path[j][-2][1], 1)
                    else:
                        blocklist2 = []
                        for a in self.arrivelist:
                            blocklist2.append(elist[a])
                        if len(replan) > 0:
                            blocklist2.append(path[j][0])
                        if len(replan2) > 0 :
                            blocklist2.append(path[j][0])
                        while True:
                            mapsize = (self.width, self.height)
                            if len(result_path[j]) > 0:
                                pos_cnode = result_path[j][-1]
                            else:
                                pos_cnode = slist[j]
                            pos_enode = elist[j]
                            myAstar = AStar(mapsize, pos_cnode, pos_enode)
                            #print("规划的起点",pos_cnode)
                            blocklist2 = blocklist2 + fixed_block
                            myAstar.setBlock(blocklist2)
                            #print("障碍",blocklist2)
                            routelist = [] #记录搜索到的最优路径
                            if myAstar.run() == 1:
                                routelist = myAstar.get_minroute()
                                #print("重新规划出的路径", routelist)
                                #相向冲突检测
                                #和前面对比
                                for f in range(0, j):
                                    if len(result_path[f]) > 1:
                                        #print("result_path[f][-2]",result_path[f][-2])
                                        #print("result_path[f][-1]",result_path[f][-1])
                                        if result_path[f][-2] == routelist[1] and result_path[f][-1] == routelist[0]:#发生相向冲突
                                            grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                                for f in range(j):
                                    if len(path[f]) > 0:
                                        if result_path[f][-2] == routelist[1] and result_path[f][-1] == routelist[0]:#发生相向冲突
                                            grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)     
                                for f in range(j + 1, self.n):
                                    if len(result_path[f]) > 1:
                                        #print("result_path[f][-2]",result_path[f][-2])
                                        #print("result_path[f][-1]",result_path[f][-1])
                                        if result_path[f][-2] == routelist[1] and path[f][0] == routelist[0]:#发生相向冲突
                                            grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                                for f in range(j + 1, self.n):
                                    if len(result_path[f]) > 1 and len(path[f]) > 0:
                                        if result_path[f][-1] == routelist[1] and path[f][0] == routelist[0]:#发生相向冲突
                                            grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                                if grid_map_collisions.get_value_from_xy_index(routelist[1][0], routelist[1][1]) == 0:#按重新规划的走
                                    result_path[j].append(routelist[1])
                                    grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                                    path[j] = routelist[2:]
                                    break
                                else:
                                    blocklist2.append(routelist[1])
                            else:
                                print('路径规划失败！')
                                flat = 0
                                break
                        #避免交换位置的相向冲突
                        for k in range(j + 1, self.n):
                            if len(path[j]) > 1 and len(path[k]) > 1:
                                if path[j][0] == path[k][1] and path[j][1] == path[k][0]:
                                    replan.append(k)
                                    grid_map_collisions.set_value_from_xy_index(path[j][1][0],path[j][1][1], 1)
                            if len(path[j]) > 1 and len(result_path[k]) > 1:
                                if path[j][0] == result_path[k][-1] and path[k][0] == result_path[j][-2]:
                                    replan2.append(k)
                                    grid_map_collisions.set_value_from_xy_index(result_path[j][-2][0],result_path[j][-2][1], 1)  
                elif value == 2:#有到达终点的点，重新规划路径
                    print("重新规划")
                    while True:
                        mapsize = (self.width, self.height)
                        pos_cnode = result_path[j][-1]
                        pos_enode = elist[j]
                        myAstar = AStar(mapsize, pos_cnode, pos_enode)
                        blocklist = blocklist + fixed_block
                        myAstar.setBlock(blocklist)
                        routelist = [] #记录搜索到的最优路径
                        if myAstar.run() == 1:
                            routelist = myAstar.get_minroute()
                            #print("重新规划的路径：",routelist)
                            #相向冲突检测
                            #和前面对比
                            for f in range(0, j):
                                if len(result_path[f]) > 1:
                                    #print("result_path[f][-2]",result_path[f][-2])
                                    #print("result_path[f][-1]",result_path[f][-1])
                                    if result_path[f][-2] == routelist[1] and result_path[f][-1] == routelist[0]:#发生相向冲突
                                        grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                            for f in range(j + 1, self.n):
                                #print("result_path[f][-1]",result_path[f][-1])
                                #print("path[f][0]",path[f][0])
                                if len(result_path[f]) >= 1 and len(path[f]) > 0:
                                    #print("result_path[f][-1]",result_path[f][-1])
                                    #print("path[f][0]",path[f][0])
                                    if result_path[f][-1] == routelist[1] and path[f][0] == routelist[0]:#发生相向冲突
                                        grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                            for f in range(j+1, self.n):
                                if len(result_path[f]) > 1:                                   
                                    #print("result_path[f][-2]",result_path[f][-2])
                                    #print("result_path[f][-1]",result_path[f][-1])
                                    if result_path[f][-2] == routelist[1] and result_path[f][-1] == routelist[0]:#发生相向冲突
                                        grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                            if grid_map_collisions.get_value_from_xy_index(routelist[1][0], routelist[1][1]) == 0:#按重新规划的走
                                result_path[j].append(routelist[1])
                                path[j] = routelist[2:]
                                grid_map_collisions.set_value_from_xy_index(routelist[1][0],routelist[1][1], 1)
                                break
                            else:
                                blocklist.append(routelist[1])
                        else:
                            print('路径规划失败！')
                            flat = 0
                            break
                        #避免交换位置的相向冲突
                        for k in range(j + 1, self.n):
                            if len(path[j]) > 1 and len(path[k]) > 1:
                                if path[j][0] == path[k][1] and path[j][1] == path[k][0]:
                                    replan.append(k)
                                    grid_map_collisions.set_value_from_xy_index(path[j][1][0],path[j][1][1], 1)
                            if len(path[j]) > 1 and len(result_path[k]) > 1 and len(path[k]) > 1 :
                                if path[j][0] == result_path[k][-1] and path[k][0] == result_path[j][-1]:
                                    replan2.append(k)
                                    grid_map_collisions.set_value_from_xy_index(result_path[j][-1][0],result_path[j][-1][1], 1)
                #print("重新规划路径的点",replan)
                #print("重新规划路径的点2",replan2)
            t = t + 1
            #print("t",t)
            if flat < 1:
                break            
            #print("当前",result_path)
            #print("初始路径",path)
            #print("已到达",self.arrivelist)

            #判断退出循环的条件
            if len(self.arrivelist) == self.n:
                #print(result_path)
                flat = 1
                break
        return result_path, t, flat

    def draw_paths(self,result_path):
        plt.figure()
        t = len(result_path[0])

        for i in range(0,self.n):
            x = []
            y = []
            for j in range(t):
                x.append(result_path[i][j][0])
                y.append(result_path[i][j][1])
            plt.plot(x, y)
        plt.grid(ls="--")
        plt.xticks(range(1,self.width),fontsize=12)  
        plt.yticks(range(1,self.height),fontsize=9)
        plt.show()

    def draw_paths_animation2(self, arr_start_index, arr_end_index,result_path):
        colorlist = []
        t = len(result_path[0])
        for i in range(self.n):
            #给每辆车规定颜色
            r = randint(0, 256)
            g = randint(0, 256)
            b = randint(0, 256)
            color = (r/256, g/256, b/256)
            colorlist.append(color)
        for j in range(t):
            #画固定障碍
            for block in self.block:
                x, y = self.calc_xyindex(block)
                plt.plot(x + 0.5, y + 0.5,marker="s",markersize=10,color="black" )
            for i in range(self.n):
                #转为二维坐标
                pos_snode = tuple(self.calc_xyindex(arr_start_index[i]))
                pos_enode = tuple(self.calc_xyindex(arr_end_index[i]))
                plt.plot(pos_snode[0] + 0.5, pos_snode[1] + 0.5, "^r")
                plt.text(pos_snode[0] + 0.5, pos_snode[1] + 0.5, str(i), ha='left',  wrap=True)
                plt.plot(pos_enode[0] + 0.5, pos_enode[1] + 0.5, "^b")
                plt.text(pos_enode[0] + 0.5, pos_enode[1] + 0.5, str(i), ha='left',  wrap=True)
            plt.axis("equal")
            plt.xticks(range(0,self.width + 1),fontsize=12)  
            plt.yticks(range(0,self.height + 1),fontsize=9)
            plt.grid(ls="--")
            #plt.grid(ls="-")
            for i in range(self.n):
                rx = []
                ry = []
                for m in range(j + 1 ):
                    rx.append(result_path[i][m][0] + 0.5)
                    ry.append(result_path[i][m][1] + 0.5)
                plt.plot(rx, ry, color = colorlist[i])
            count = 0  
  
            # 创建自定义的文件名格式  
            custom_name = "my_image_"  
  
            # 指定图片保存路径  
            figure_save_path = "exam1"  
            if not os.path.exists(figure_save_path):  
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建  
  
            # 循环直到找到一个唯一的文件名  
            while True:  
                # 生成基于时间的文件名  
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  
                file_name = custom_name + timestamp + str(count) + '.png'  
      
                # 检查文件是否已存在  
                if not os.path.exists(os.path.join(figure_save_path, file_name)):  
                    break  # 如果文件不存在，跳出循环  
                else:  
                    count += 1  # 如果文件已存在，增加计数器并继续循环  
  
            # 保存图片，使用唯一命名的文件名  
            plt.savefig(os.path.join(figure_save_path, file_name)) 
            plt.show()   
