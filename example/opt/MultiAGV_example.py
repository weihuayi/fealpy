
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
import datetime
import os
from random import randint
import sys
import time 
from fealpy.backend import backend_manager as bm
from fealpy.opt.A_star import AStar, GridMap, Graph
# from fealpy.iopt.A_star import AStar, GridMap, Graph
bm.set_backend('pytorch')


def test(times, n, height, width, block_num):
    """
    k:测试次数
    """
    start_time = time.perf_counter()
    run_count = 0
    success_count = 0
    Turn_sum = []
    Turn_avr = []
    Turn_max = []
    while run_count < times:
        G1 = Graph(n , height, width)
        G1.block = [29, 30, 9, 21] # 障碍
        arr_start_index =   [1, 14, 26, 12, 6, 2, 31] 
        arr_end_index =  [8, 23, 35, 15, 34, 13, 28]
        print('起点和终点列表', arr_start_index, arr_end_index)
        run_count += 1
        print('运行次数', run_count)
        sorted_arr_start_index, sorted_arr_end_index = G1.sort_by_distance(arr_start_index, arr_end_index)
        print('排序后起点和终点列表',sorted_arr_start_index.tolist(), sorted_arr_end_index.tolist())
        k = 1
        result_path, t, flat = G1.paths_avoid_collisions(sorted_arr_start_index.tolist(), sorted_arr_end_index.tolist(), k)
        print(result_path)
        if flat > 0:
            result_path_index = G1.cal_result_path_index(result_path, t)
            print('最终线路',result_path_index)
            turn = G1.turn_num(result_path, t - 2)
            turn_max = bm.max(turn)
            turn_avr = bm.mean(turn)
            turn_sum = bm.sum(turn)
            Turn_sum.append(turn_sum)
            Turn_avr.append(turn_avr)
            Turn_max.append(turn_max)
            print('转弯次数',turn)
            G1.check_collision_index(result_path, t - 2)
            #G1.draw_paths_animation2(arr_start_index, arr_end_index,result_path)
        success_count = success_count + flat
        print('成功次数', success_count)
    print('每次测试最大转弯数', Turn_max)
    print('每次测试平均转弯数', Turn_avr)
    print('每次测试总转弯数', Turn_sum)
    print('平均最大转弯数', sum(Turn_max) / len(Turn_max))
    print('平均转弯数', sum(Turn_avr) / len(Turn_avr))
    print('平均总转弯数', sum(Turn_sum) / len(Turn_sum))
    end_time = time.perf_counter()
    running_time = end_time - start_time
    print("Runtime: ", running_time)
    # G1.draw_paths(result_path)
    # G1.draw_paths_animation2(arr_start_index, arr_end_index,result_path)


test(1, 7, 6, 6, 5)
