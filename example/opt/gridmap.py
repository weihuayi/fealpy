
from functools import total_ordering
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

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
               
    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("n_data:", self.n_data)    
        
    def show_grid_data(self):
        float_data_array = np.array([d for d in self.data])
        grid_data = np.reshape(float_data_array, (self.height, self.width))
        return grid_data        

    
