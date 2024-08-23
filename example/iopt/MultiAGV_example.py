# from gridmap_2 import GridMap
# from A_star import AStar, GridMap

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
import datetime
import os
from random import randint
import sys
import time
from fealpy.experimental.backend import backend_manager as bm
from fealpy.iopt.A_star import AStar, GridMap

class Graph:
    esp = 0.001
    def __init__(self, n, width, height ):
        """
        n:    小车数量
        size: 地图大小为width * height的网格
        """
        self.n = n
        self.width = width
        self.height = height
        self.grid_map = GridMap(self.width, self.height)
        self.arrivelist = []#到达列表，固定障碍
        self.direction = np.zeros((1,self.n))
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
        return np.array(arr_start_index), np.array(arr_end_index)    

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
        return np.array(sorted_arr_start_index), np.array(sorted_arr_end_index)     
                
    
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
                #showresult(mapsize, pos_snode, pos_enode, self.block, routelist)
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
                #print(routelist)

                if abs(pos_cnode[0] - pos_enode[0]) < esp and abs(pos_cnode[1] - pos_enode[1]) < esp:
                    break
            #print(routelist)
            #showresult(mapsize, pos_snode, pos_enode, blocklist, routelist)
            route.append(routelist)
        return route
        
    
    def check_collision_index(self, result_path, t):

        #传入某个时间点的不同点所在坐标数组
        esp = 0.0001    
        collision_message = np.zeros((1,6))
        paths_index = self.cal_result_path_index(result_path, t + 2)
     
        for t in range(0,t):
            result = paths_index[:,t]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    
                    #冲突点在节点的情况
                    if abs(result[i] - result[j]) < esp :
                        print("时间为",t,"时",i, "与", j, "相撞","坐标为",self.calc_xyindex(result[j]))
                        collision = np.array([int(t), int(i), int(j), result[j], 0, 0])
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[j])[0],self.calc_xyindex(result[j])[1], 7.0)
                    
                        #判断碰撞类型
                        a_previous = paths_index[int(i), t-1]
                        a_next = paths_index[int(i), t+1]
    
                        b_previous = paths_index[int(j), t-1]
                        b_next = paths_index[int(j),  t+1]
                
                        if abs(a_previous - b_next) < esp and abs(b_previous - a_next) < esp :
                    
                            print("相向冲突:",i,j)
                            collision[-1] = 1
       
                        else:
                            print("节点冲突:",i,j)
                
                        collision = collision.reshape(1, -1)
                        collision_message = np.concatenate((collision_message, collision), axis=0)
                        
                      #冲突点不在节点的情况
                    if abs(result[i] - paths_index[j][t+1]) < esp and abs(result[j] - paths_index[i][t+1]) < esp:
                        print("时间为",t,"时",i, "与", j, "相撞","坐标为",self.calc_xyindex(result[i]),self.calc_xyindex(result[j]) )
                        print("相向冲突:",i,j)
                        
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[j])[0],self.calc_xyindex(result[j])[1], 7.0)
                        self.grid_map.set_value_from_xy_index(self.calc_xyindex(result[i])[0],self.calc_xyindex(result[i])[1], 7.0)
                        collision = np.array([int(t), int(i), int(j), result[i], result[j],0])
                        collision = collision.reshape(1, -1)
                        collision_message = np.concatenate((collision_message, collision), axis=0)    
        #self.grid_map.plot_grid_map()
        return collision_message[1:]
    
    def turn_num(self, result_path, t):
        """
        计算不同点拐弯次数
        """
        esp = 0.001
        turn = np.zeros((1,self.n))
        paths_index = self.cal_result_path_index(result_path, t + 2)
        for i in range(self.n):
            for j in range(1,t):
                x_previous, y_previous = self.calc_xyindex(paths_index[i][j - 1])
                x_next, y_next = self.calc_xyindex(paths_index[i][j + 1])
                if x_previous != x_next and y_previous != y_next :
                    turn[:,i] = turn[:,i] + 1
        return turn
    
    def cal_result_path_index(self,result_path, t):
        result_path_index = np.zeros((self.n, t ))
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
                                    
                                        #print("path[f][0]",path[f][0])
                                        #print("result_path[f][-1]",result_path[f][-1])
                       
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


def test(times, n , height, width, block_num):
    """
    k:测试次数
    """
    run_count = 0
    success_count = 0
    Turn_sum = []
    Turn_avr = []
    Turn_max = []
    while run_count < times:
        G1 = Graph(n , height, width)
        # G1.generate_block(block_num)#随机生成障碍
        # arr_start_index, arr_end_index = G1.set_data_index()
        G1.block = [29, 30, 9, 21]
        arr_start_index =   [1, 14, 26, 12, 6, 2, 31] 
        arr_end_index =  [8, 23, 35, 15, 34, 13, 28]
        print('起点和终点列表',arr_start_index, arr_end_index)
        run_count += 1
        # print('******'*10)
        print('运行次数',run_count)
        sorted_arr_start_index, sorted_arr_end_index = G1.sort_by_distance(arr_start_index, arr_end_index)
        print('排序后起点和终点列表',sorted_arr_start_index, sorted_arr_end_index)
        k = 1
        #  result_path, t, flat = G1.paths_avoid_collisions( arr_start_index, arr_end_index, k)
        result_path, t, flat = G1.paths_avoid_collisions( sorted_arr_start_index, sorted_arr_end_index, k)
        if flat > 0:
            result_path_index = G1.cal_result_path_index(result_path, t)
            print('最终线路',result_path_index)
            turn = G1.turn_num(result_path,t - 2)
            turn_max = np.max(turn)
            turn_avr = np.mean(turn)
            turn_sum = np.sum(turn)
            Turn_sum.append(turn_sum)
            Turn_avr.append(turn_avr)
            Turn_max.append(turn_max)
            print('转弯次数',turn)
            G1.check_collision_index(result_path, t - 2)
            #G1.draw_paths_animation2(arr_start_index, arr_end_index,result_path)
        success_count = success_count + flat
        print('成功次数',success_count)
    print('每次测试最大转弯数',Turn_max)
    print('每次测试平均转弯数',Turn_avr)
    print('每次测试总转弯数',Turn_sum)
    print('平均最大转弯数',sum(Turn_max)/len(Turn_max))
    print('平均转弯数',sum(Turn_avr)/len(Turn_avr))
    print('平均总转弯数',sum(Turn_sum)/len(Turn_sum))

    # G1.draw_paths(result_path)
    # G1.draw_paths_animation2(arr_start_index, arr_end_index,result_path)

start_time = time.perf_counter()
test(1, 7, 6, 6, 5)
end_time = time.perf_counter()
running_time = end_time-start_time
print("Runtime: ", running_time)