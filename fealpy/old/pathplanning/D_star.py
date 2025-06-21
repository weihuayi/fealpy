"""

D* grid planning


"""
import math


from sys import maxsize

import matplotlib.pyplot as plt
import datetime
import os

show_animation = True


class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state

        self.h = 0#每个点有h和k两个代价
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        #return math.sqrt(math.pow((self.x - state.x), 2) +
        #                 math.pow((self.y - state.y), 2))  #欧式距离
        return abs(self.x - state.x) + abs(self.y -  state.y)

    def set_state(self, state):
        """
        设置状态
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
     
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if i != 0 and j != 0:#横平竖直走
                    continue

                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])

        return state_list

    def set_obstacle(self, point_list):
        """
        设置障碍
        point_list:障碍列表

        """
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")


class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()

    def process_state(self):
        """
        处理节点信息
        第一阶段，环境障碍物已知，都是静态
        """
        #获取最小k的点
        x = self.min_state()

        if x is None:
            return -1
        #获取最小k值
        k_old = self.get_kmin()
     

        #移除
        self.remove(x)

        if k_old < x.h:
            #节点受到障碍的影响,h值变大
            for y in self.map.get_neighbors(x):
                #判断邻居节点，看是否让h减小
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    #经过y,走到x会让代价减少
                    x.parent = y
                    x.h = y.h + x.cost(y)#置换之前的代价


        if k_old == x.h:
            #可以减少到k_old的情况
            for y in self.map.get_neighbors(x):
                #判断邻居节点是否有必要作为父节点
                """
                三种情况：
                1、Y还没在openlist里面，以X为父节点
                2、父节点X的代价有更新过，可能由障碍引起
                3、Y的父节点不是X，但Y可以通过将X作为父节点使Y的代价减少
                
                A*算法只有第一种情况
                """
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            #X受到影响，未恢复到Lower态 
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        #经过x的代价更小，x入队
                        self.insert(x, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            #y入队 
                            self.insert(y, y.h)
      
        return self.get_kmin()

    def min_state(self):
        """
        在open_list中找到最小k的点
        """
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        """
        取最小的k值
        """
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        """
        open_list 插入
        """
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        """
        从open_list中移除
        """
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        """
        要用于修正受障碍物影响导致代价发生变化的节点信息
        第二阶段 动态避障，充分利用第一阶段保存的信息  
        """
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end, blocklist):



        rx = []
        ry = []

        #终点入队
        self.insert(end, 0.0)

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")#标记起点
        s = start
        s = s.parent
        s.set_state("e")#标记终点
        tmp = start

        new_ox, new_oy = AddNewObstacle(self.map) # add new obstacle after the first search finished

        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)


            if show_animation:
                plt.plot(start.x + 0.5, start.y + 0.5, "^r")
                plt.plot(end.x + 0.5, end.y + 0.5, "^b")

            #画固定障碍
                for block in blocklist:
                    plt.plot(block[0] + 0.5, block[1] + 0.5,marker="s",markersize=12,color="black" )

                for i in range(len(new_ox)):
                    plt.plot(new_ox[i] + 0.5, new_oy[i] + 0.5, marker="s",markersize=12,color="green")

                plt.axis("equal")
                plt.xticks(range(0,self.map.row + 1),fontsize=12)  
                plt.yticks(range(0,self.map.col + 1),fontsize=9)
                plt.grid(ls="--")

                new_rx = [x + 0.5 for x in rx] 
                new_ry = [y + 0.5 for y in ry] 
                plt.plot(new_rx, new_ry, "-r")
                plt.pause(0.01)
                
                mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
            
                # 指定图片保存路径
                
                figure_save_path = "exam5"
                if not os.path.exists(figure_save_path):
                    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
                plt.savefig(os.path.join(figure_save_path , mkfile_time))#创建文件夹，储存有时间命名的多张图片  
                
            if tmp.parent.state == "#":
                #遇到障碍，调整
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("e")


        #画图
        plt.plot(start.x + 0.5, start.y + 0.5, "^r")
        plt.plot(end.x + 0.5, end.y + 0.5, "^b")

        #画固定障碍
        for block in blocklist:
            plt.plot(block[0] + 0.5, block[1] + 0.5,marker="s",markersize=12,color="black" )

        for i in range(len(new_ox)):
            plt.plot(new_ox[i] + 0.5, new_oy[i] + 0.5, marker="s",markersize=12,color="green")
        rx.append(end.x)
        ry.append(end.y)
        rx = [num + 0.5 for num in rx]
        ry = [num + 0.5 for num in ry]
        plt.axis("equal")
        plt.xticks(range(0,self.map.row + 1),fontsize=12)  
        plt.yticks(range(0,self.map.col + 1),fontsize=9)
        plt.grid(ls="--")
        plt.plot(rx , ry , "-r")
      
        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
            
        # 指定图片保存路径
                
        figure_save_path = "exam5"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path , mkfile_time))#创建文件夹，储存有时间命名的多张图片  
        return rx, ry

    def modify(self, state):
        #动态避障
        self.modify_cost(state)
        while True:
            #利用之前的信息重新规划
            k_min = self.process_state()
            if k_min >= state.h:
                break

def AddNewObstacle(map:Map):
    ox, oy = [], []
    """
    for i in range(5, 21):
        ox.append(i)
        oy.append(40)
    map.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    """
    ox = [12,12,12,12,12,12]
    oy = [13,16,15,18,17,8]
    map.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    if show_animation:
        plt.pause(0.001)
        for i in range(len(ox)):
            plt.plot(ox[i] + 0.5, oy[i] + 0.5, marker="s",markersize=12,color="green")
    return ox, oy

def main():
    m = Map(20, 20)
    ox, oy = [], []

    #m.set_obstacle([(i, j) for i, j in zip(ox, oy)])
    #地图中设置障碍
    blocklist = [(10, 18), (3, 0), (2, 1), (6, 12), (1, 18), (5, 14), (16, 4), (17, 9), (18, 18), (5, 8), (18, 5), (6, 7), (8, 5), (16, 5), (11, 17), (4, 12), (7, 0), (0, 8), (2, 16), (8, 4), (6, 19), (3, 14), (7, 13), (19, 15), (10, 3), (9, 5), (11, 6), (0, 10), (10, 7), (19, 4), (10, 1), (12, 13), (18, 11), (12, 4), (12, 18), (11, 3), (8, 3), (13, 18), (5, 17), (4, 13), (4, 2), (19, 14), (2, 4), (8, 10), (5, 7), (19, 6), (6, 1), (15, 16), (12, 6), (13, 1), (10, 2), (4, 14), (7, 17), (15, 13), (15, 0), (6, 3), (7, 7), (3, 2), (15, 11), (14, 12), (7, 19), (12, 14), (13, 12), (16, 15), (15, 2), (5, 9), (7, 1), (14, 14), (13, 2), (13, 10)]
    m.set_obstacle(blocklist)




    #start = [10, 20]
    #goal = [50, 50]

    start = [0,0]
    goal = [14,18]



    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.axis("equal")

    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]
    dstar = Dstar(m)
    rx, ry = dstar.run(start, end, blocklist)

    rx.append(end.x)
    ry.append(end.y)
 
    new_rx = [x + 0.5 for x in rx] 
    new_ry = [y + 0.5 for y in ry] 

    #print("rx", new_rx)
    #print("ry",new_ry)
   
    if show_animation:
        plt.plot(new_rx, new_ry, "-r")
      


if __name__ == '__main__':
    main()