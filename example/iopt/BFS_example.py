import matplotlib.pyplot as plt
from gridmap import GridMap
import datetime
import matplotlib.animation as animation

show_animation = True

class Breath_frist_search:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_map = GridMap(self.width, self.height)#初始化地图
        self.result_path = []
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index, parent):           
            self.x = x  
            self.y = y  
            self.cost = cost
            self.parent_index = parent_index
            self.parent = parent

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index) 

    def calc_xyindex(self, index):
        #返回编号的x, y坐标
        y_ind, x_ind = divmod(index, self.width)
        return x_ind, y_ind

    def calc_grid_index(self, x, y):
        #返回x,y坐标对应的编号
        return int(y * self.width + x)   
        

    def planning(self, sx, sy, gx, gy, blocklist):
        nstart = self.Node(sx, sy, 0.0, -1, None)
        ngoal = self.Node(gx, gy, 0.0, -1, None)
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart.x, nstart.y)] = nstart
        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break
            current = open_set.pop(list(open_set.keys())[0])
            c_id = self.calc_grid_index(current.x, current.y)#记录当前节点的编号
            closed_set[c_id] = current

            # show graph
            if show_animation:  
                #画起点和终点
                plt.plot(sx + 0.5, sy + 0.5, "^r")
                plt.plot(gx + 0.5, gy + 0.5, "^b")

                #画固定障碍
                for block in blocklist:
                    plt.plot(block[1] + 0.5, block[0] + 0.5,marker="s",markersize=15,color="black" )

                plt.axis("equal")
                plt.xticks(range(0,self.width + 1),fontsize=12)  
                plt.yticks(range(0,self.height + 1),fontsize=9)
                plt.grid(ls="--")
                plt.plot(current.x + 0.5, current.y + 0.5 , "xc")
                # plt.show() 
                mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:#找到终点
                print("Find goal")
                ngoal.parent_index = current.parent_index#找父节点
                ngoal.cost = current.cost
                break

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id, None)
                n_id = self.calc_grid_index(node.x, node.y)

                if not self.verify_node(node, blocklist):
                    continue

                if (n_id not in closed_set) and (n_id not in open_set):
                    node.parent = current
                    open_set[n_id] = node#进入队列

        rx, ry = self.calc_final_path(ngoal, closed_set)#回溯
        for i in range(len(rx)):
            self.result_path.append((rx[i], ry[i]))
        self.result_path.reverse()

        if show_animation:  
            rx = [num + 0.5 for num in rx]
            ry = [num + 0.5 for num in ry]
            plt.plot(rx , ry , "-r")
            plt.pause(0.01)
            plt.show()
            mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
        return self.result_path
    

    def calc_final_path(self, ngoal, closedset):
        # 生成路径
        rx, ry = [ngoal.x], [ngoal.y]
        n = closedset[ngoal.parent_index]
        while n is not None:
            rx.append(n.x)
            ry.append(n.y)
            n = n.parent  
        return rx, ry
    
    def verify_node(self, node, blocklist):

        #判断点有无超出范围
        px = node.x
        py = node.y
        if px < 0:
            return False
        elif py < 0:
            return False
        elif px >= self.width:
            return False
        elif py >= self.height:
            return False
        self.calc_obstacle_map( blocklist )
        grid_data = self.grid_map.show_grid_data()
        if grid_data[node.x][node.y] == 1:
            return False
        return True
    
    def calc_obstacle_map(self, blocklist):
        for block in blocklist:
            self.grid_map.set_value_from_xy_index(block[0], block[1], 1)

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]

        return motion

if __name__=="__main__":
    G1 = Breath_frist_search(20, 20)
    blocklist = [(10, 18), (3, 0), (2, 1), (6, 12), (1, 18), (5, 14), (16, 4), (17, 9), (18, 18), (5, 8), (18, 5), (6, 7), (8, 5), (16, 5), (11, 17), (4, 12), (7, 0), (0, 8), (2, 16), (8, 4), (6, 19), (3, 14), (7, 13), (19, 15), (10, 3), (9, 5), (11, 6), (0, 10), (10, 7), (19, 4), (10, 1), (12, 13), (18, 11), (12, 4), (12, 18), (11, 3), (8, 3), (13, 18), (5, 17), (4, 13), (4, 2), (19, 14), (2, 4), (8, 10), (5, 7), (19, 6), (6, 1), (15, 16), (12, 6), (13, 1), (10, 2), (4, 14), (7, 17), (15, 13), (15, 0), (6, 3), (7, 7), (3, 2), (15, 11), (14, 12), (7, 19), (12, 14), (13, 12), (16, 15), (15, 2), (5, 9), (7, 1), (14, 14), (13, 2), (13, 10)]
    sx = 6
    sy = 8
    gx = 9
    gy = 1
    res_path = G1.planning( sx, sy, gx, gy, blocklist)
    print(res_path)





    



















