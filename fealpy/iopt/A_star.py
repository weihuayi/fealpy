
import numpy as np
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
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                self.set_value_from_xy_index(x, y ,2.0)
                self.obstacle.append((x,y))
        else:
            print("设置的概率不在范围内")
        
    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("n_data:", self.n_data)    
        
    def plot_grid_map(self, ax=None):
        float_data_array = np.array([d for d in self.data])
        print(float_data_array)
        grid_data = np.reshape(float_data_array, (self.height, self.width))
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
        float_data_array = np.array([d for d in self.data])
        #print(float_data_array)
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        #print(grid_data)
        return grid_data        