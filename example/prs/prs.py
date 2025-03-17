import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from data import H
from load_roller_screw import *


class Roller():
    def __init__(self, center, data, order_roller_screw):
        self._data = data
        self.center = center 
        # 行星滚柱丝杠序号
        self.id_roller_screw = np.where(data[:, 0] == order_roller_screw)
        self.roller_screw_temp = data[self.id_roller_screw[0], :]
        # 滚柱牙型圆弧半径(mm)，radius_roller
        self.radius_roller = self.roller_screw_temp[0, 18-1]
        # 滚柱螺纹半牙减薄量
        self.reduce = self.roller_screw_temp[0, 29-1]
        # 滚柱螺纹削根系数
        self.cutter_bottom = self.roller_screw_temp[0, 26-1]
        # 滚柱中经
        self.D_pitch = self.roller_screw_temp[0, 45-1]
        
        # 滚柱实际根径
        self.D_bottom = self.roller_screw_temp[0, 42-1]
        # 滚柱轴长度
        self.length_shaft = self.roller_screw_temp[0, 48-1]
        # 滚柱轴外径
        self.diameter_outer = self._data[0, 51-1]
        # 滚柱轴内径
        self.diameter_inner = self.roller_screw_temp[0, 54-1]
        # 滚柱螺纹长度
        self.length_thread = self._data[0, 57-1]
        # 滚柱螺纹左端相对于滚柱轴左端的位置
        self.delta_thread = self.roller_screw_temp[0, 69-1]
        # 密度
        self.density = self.roller_screw_temp[0, 73-1] 
        # 弹性模量
        self.E = self.roller_screw_temp[0, 76-1]
        # 泊松比
        self.nu = self.roller_screw_temp[0, 79-1]
        # 轮缘轴序号
        self.shaft_self = self.roller_screw_temp[0, 133-1]
        # 综合模量
        self.E0 = self.E / (1 - self.nu ** 2)
        # 截面积
        self.area = np.pi * (self.D_bottom ** 2 - self.diameter_inner ** 2) / 4

    def add_plot(self, axes):
        """画滚柱相关的圆
        """
        r = self.diameter_inner
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')


class Screw():
    def __init__(self, data, order_roller_screw, modify_screw, norm_roller_screw):
        self._data = data
        # 丝杠侧修行向量
        modify_screw = np.array(modify_screw, dtype=np.float64)
        self.modify = (10 ** 3 * modify_screw) - np.min(10 ** 3 * modify_screw, axis=0)

        # 行星滚柱丝杠序号
        self.id_roller_screw = np.where(data[:, 0] == order_roller_screw)
        self.roller_screw_temp = data[self.id_roller_screw[0], :]
        # 螺距
        self.pitch = self.roller_screw_temp[0, 16-1]
        # 牙型角
        self.angle = self.roller_screw_temp[0, 17-1]
        # 丝杠螺纹削根系数
        self.cutter_bottom = self.roller_screw_temp[0, 22-1]
        # 丝杠螺纹半牙减薄量
        self.reduce = self.roller_screw_temp[0, 27-1]
        # 丝杠中经
        self.D_pitch = self.roller_screw_temp[0, 43-1]

        # 丝杠实际根径
        self.D_bottom = self.roller_screw_temp[0, 38-1]
        # 丝杠轴长度
        self.length_shaft = self.roller_screw_temp[0, 46-1]
        # 丝杠轴外径
        self.diameter_outer = self.roller_screw_temp[0, 49-1]
        # 丝杠轴内径
        self.diameter_inner = self.roller_screw_temp[0, 52-1]
        # 丝杠螺纹长度
        self.length_thread = self.roller_screw_temp[0, 55-1]
        # 丝杠螺纹左端相对于丝杠轴左端的位置
        self.delta_thread = self.roller_screw_temp[0, 67-1]
        # 密度
        self.density = self.roller_screw_temp[0, 71-1] 
        # 弹性模量
        self.E = self.roller_screw_temp[0, 74-1]
        # 泊松比
        self.nu = self.roller_screw_temp[0, 77-1]
        # 轮缘轴序号
        self.shaft_self = self.roller_screw_temp[0, 132-1]
        # 综合模量
        self.E0 = self.E / (1 - self.nu ** 2)
        # 截面积
        self.area = np.pi * (self.D_bottom ** 2 - self.diameter_inner ** 2) / 4
        # 丝杠第一主曲率
        self.p_1 = 0
        

        id_norm_1 = (norm_roller_screw[:, 1-1] == order_load)
        id_norm_2 = (norm_roller_screw[:, 2-1] == order_roller_screw)
        if id_norm_1 * id_norm_2 == True:
            id_norm = 0

        # 丝杠工作面
        self.face = norm_roller_screw[id_norm, 7-1]
        # 丝杠工作面压力角
        self.press_angle = np.abs(norm_roller_screw[id_norm, 8-1])
        # 法向载荷
        self.force = np.abs(norm_roller_screw[id_norm, 10-1])
        # 受载法向矢量
        self.vector_norm = norm_roller_screw[id_norm, 11-1:13][None, :]
        # 丝杠轴向分量比例
        self.ratio = np.abs(self.vector_norm[0, 2]) / np.linalg.norm(self.vector_norm)


    def add_plot(self, axes):
        """画丝杠相关的圆
        """
        r = self.diameter_inner
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')



class Nut():
    def __init__(self, data, order_roller_screw, modify_nut, norm_roller_screw):
        self._data = data
        # 螺母侧修行向量
        modify_nut = np.array(modify_nut, dtype=np.float64)
        self.modify = (10 ** 3 * modify_nut) - np.min(10 ** 3 * modify_nut, axis=0)
        # 行星滚柱丝杠序号
        self.id_roller_screw = np.where(data[:, 0] == order_roller_screw)
        self.roller_screw_temp = data[self.id_roller_screw[0], :]
        # 螺母螺纹削根系数
        self.cutter_bottom = self.roller_screw_temp[0, 24-1]
        # 螺母螺纹半牙减薄量
        self.reduce = self.roller_screw_temp[0, 28-1]
        # 螺母中经
        self.D_pitch = self.roller_screw_temp[0, 44-1]

        # 螺母实际根径
        self.D_bottom = self.roller_screw_temp[0, 40-1]
        # 螺母轴长度
        self.length_shaft = self.roller_screw_temp[0, 47-1]
        # 螺母轴外径
        self.diameter_outer = self.roller_screw_temp[0, 50-1] 
        # 螺母轴内径
        self.diameter_inner = self._data[0, 53-1]
        # 螺母螺纹长度
        self.length_thread = self._data[0, 56-1]
        # 螺母螺纹左端相对于螺母轴左端的位置
        self.delta_thread = self.roller_screw_temp[0, 68-1]

        # 密度
        self.density = self.roller_screw_temp[0, 72-1] 
        # 弹性模量
        self.E = self.roller_screw_temp[0, 75-1]
        # 泊松比
        self.nu = self.roller_screw_temp[0, 78-1]
        # 轮缘轴序号
        self.shaft_self = self.roller_screw_temp[0, 131-1]
        # 综合模量
        self.E0 = self.E / (1 - self.nu ** 2)
        # 截面积
        self.area = np.pi * (self.diameter_outer ** 2 - self.D_bottom ** 2) / 4

        id_norm_1 = (norm_roller_screw[:, 1-1] == order_load)
        id_norm_2 = (norm_roller_screw[:, 2-1] == order_roller_screw)
        if id_norm_1 * id_norm_2 == True:
            id_norm = 0

        # 工作面
        self.face = norm_roller_screw[id_norm, 15-1]
        # 工作面压力角
        self.press_angle = np.abs(norm_roller_screw[id_norm, 16-1])
        # 法向载荷
        self.force = np.abs(norm_roller_screw[id_norm, 18-1])
        # 受载法向矢量
        self.vector_norm = norm_roller_screw[id_norm, 19-1:21][None, :]
        # 丝杠轴向分量比例
        self.ratio = np.abs(self.vector_norm[0, 2]) / np.linalg.norm(self.vector_norm)

    def add_plot(self, axes):
        # 画螺母内啮合圆
        r = self.diameter_outer 
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        axes.plot(x, y, 'b-')



class PlanetaryRollerScrew():

    def __init__(self, order_roller_screw, data, mesh_outer_data, mesh_inner_data, norm_roller_screw, modify_nut, modify_screw, type_load):
        self._order_roller_screw = order_roller_screw
        self._data = data
        self._mesh_outer_data = mesh_outer_data
        self._mesh_inner_data = mesh_inner_data
        self.norm_roller_screw = norm_roller_screw
        self.type_load = type_load

        # 行星滚柱丝杠序号
        self.id_roller_screw = np.where(data[:, 0] == order_roller_screw)
        self.roller_screw_temp = data[self.id_roller_screw[0], :]

        self.screw = Screw(self._data, order_roller_screw, modify_screw, norm_roller_screw)
        self.nut = Nut(self._data, order_roller_screw, modify_nut, norm_roller_screw)
        self.roller = Roller(np.array([13, 0]), data, order_roller_screw)
        self.rollers = []
        # 滚柱的个数
        self.number_roller = int(self.roller_screw_temp[0, 3-1])
        # 第一个滚柱与 x 轴的夹角
        self.angle_roller = self._data[0, 13-1]
        # 第一个滚柱在全局极坐标系半径
        self.center_roller = self._mesh_inner_data[0, 3] - self._mesh_inner_data[0, 8]

        # 生成所有的滚柱
        for i in range(self.number_roller): 
            self.rollers.append(Roller(np.array([
                                                 self.center_roller * np.cos(i * (2 * np.pi) / self.number_roller), 
                                                 self.center_roller * np.sin(i * (2 * np.pi) / self.number_roller)
                                                 ]), 
                                       self._data, order_roller_screw))
        

        # 滚柱丝杠类型(1表示标准式，2表示反向式)
        self.type_roller_screw = self.roller_screw_temp[0, 1]

        # 螺距
        self.pitch_screw = self.roller_screw_temp[0, 16-1]


        # 内啮合的有效螺纹长度
        self.length_mesh_inner = self.roller_screw_temp[0, 58-1]
        # 外啮合的有效螺纹长度
        self.length_mesh_outer = self.roller_screw_temp[0, 59-1]
        # 螺母轴左端相对于丝杠轴左端的位置
        self.delta_n_s = self.roller_screw_temp[0, 65-1]
        # 滚柱轴左端相对于丝杠轴左端的位置
        self.delta_r_s = self.roller_screw_temp[0, 66-1]

        # 外啮合半径
        id_mesh_outer_1 = (mesh_outer_data[:, 0] == order_roller_screw)
        id_mesh_outer_2 = (mesh_outer_data[:, 1] == self.screw.face)
        id_mesh_outer = id_mesh_outer_1 * id_mesh_outer_2
        # 丝杠
        self.screw.radius = mesh_outer_data[id_mesh_outer, 6]
        # 滚柱
        self.roller.radius_outer = mesh_outer_data[id_mesh_outer, 9-1]
        
        # 内啮合
        id_mesh_inner_1 = (mesh_inner_data[:, 0] == order_roller_screw)
        id_mesh_inner_2=(mesh_inner_data[:, 1] == self.nut.face)
        id_mesh_inner = (id_mesh_inner_1 * id_mesh_inner_2)
        # 螺母
        self.nut.radius = mesh_inner_data[id_mesh_inner, 7-1]
        # 滚柱
        self.roller.radius_inner = mesh_inner_data[id_mesh_inner, 9-1]

        # 滚柱外啮合曲率
        # 第一主曲率
        self.roller.p_outer_1 = 1 / self.roller.radius_roller
        # 第二主曲率
        self.roller.p_outer_2 = np.cos(np.pi / 2 - self.screw.press_angle) / self.roller.radius_outer
        # 丝杠外啮合第二曲率
        self.screw.p_2 = np.cos(np.pi / 2 - self.screw.press_angle) / self.screw.radius
        # 外啮合曲率和
        self.p_outer_sum = self.roller.p_outer_1 + self.roller.p_outer_2 + self.screw.p_1 + self.screw.p_2
        # 外啮合主曲率函数
        self.p_outer_f = (np.abs(self.roller.p_outer_1-self.roller.p_outer_2) + np.abs(self.screw.p_1 - self.screw.p_2)) / self.p_outer_sum
        # 外啮合综合模量影响系数
        self.E0_outer = (1.137 * 10 ** 5 * (1 / self.screw.E0 + 1 / self.roller.E0)) ** (1 / 3)    
        # 外啮合压缩势能辅助值，B_outer
        self.B_outer = (0.25 * (self.roller.p_outer_1 + self.roller.p_outer_2) + 
                        (self.screw.p_1 + self.screw.p_2) - 
                        np.abs(self.roller.p_outer_1 - self.roller.p_outer_2) - 
                        np.abs(self.screw.p_1 - self.screw.p_2))
        
        # 滚柱内啮合曲率
        # 第一主曲率
        self.roller.p_inner_1 = 1 / self.roller.radius_roller
        # 第二主曲率
        self.roller.p_inner_2 = np.cos(np.pi / 2 - self.nut.press_angle) / self.roller.radius_inner

        # 内啮合螺母第一主曲率
        self.nut.p_1 = 0
        # 内啮合螺母第二主曲率
        self.nut.p_2 = -1 * np.cos(np.pi / 2 - self.nut.press_angle) / self.nut.radius
        # 内啮合曲率之和
        self.p_inner_sum = self.roller.p_inner_1 + self.roller.p_inner_2 + self.nut.p_1 + self.nut.p_2
        # 内啮合主曲率函数
        self.p_inner_f = (np.abs(self.roller.p_inner_1 - self.roller.p_inner_2) + np.abs(self.nut.p_1 - self.nut.p_2)) / self.p_inner_sum
        # 内啮合综合模量影响系数，E0_inner
        self.E0_inner = (1.137 * 10 ** 5 * ( 1 / self.nut.E0 + 1/ self.roller.E0)) ** (1 / 3)
        # 内啮合压缩势能辅助值，B_inner
        self.B_inner = (0.25 * (self.roller.p_inner_1 + self.roller.p_inner_2) + 
                        (self.nut.p_1 + self.nut.p_2) - 
                        np.abs(self.roller.p_inner_1 - self.roller.p_inner_2) - 
                        np.abs(self.nut.p_1 - self.nut.p_2))
        
        # 计算理论受载螺纹牙数，number
        self.number_y = 1 + np.floor(np.minimum(self.length_mesh_inner, self.length_mesh_outer) / self.pitch_screw)

        # 计算载荷作用点
        self.number_load = self.number_y * self.number_roller

        # 计算行星滚柱丝杠总的轴向载荷(N)，force_axial
        self.force_axial = ((self.type_roller_screw == 1) * (np.abs(self.nut.force) * self.nut.ratio * self.number_load) + 
                            (self.type_roller_screw == 2) * (np.abs(self.screw.force) * self.screw.ratio * self.number_load))
        
        # 有效螺纹长度
        self.length_mesh = min([self.length_mesh_inner, self.length_mesh_outer])
        # 有效螺纹啮合齿数
        self.number =  1 + int(self.length_mesh / self.pitch_screw)
        


    def flexible_roller_screw(self):
        # 计算螺纹高度
        self.high_thread = (0.5 * self.screw.pitch) / np.tan(0.5 * self.screw.angle)
        # 计算螺母中径牙厚
        self.nut.b = 0.5 * self.screw.pitch - 2 * self.nut.reduce
        # 计算螺母底径牙厚
        self.nut.a = self.screw.pitch * (1 - self.nut.cutter_bottom) - 2 * self.nut.reduce
        # 计算螺母牙根高
        self.nut.c = (0.5 - self.nut.cutter_bottom) * self.high_thread
        # 计算滚柱中径牙厚
        self.roller.b = 0.5 * self.screw.pitch - 2 * self.roller.reduce
        # 计算滚柱底径牙厚
        self.roller.a = self.screw.pitch * (1 - self.roller.cutter_bottom) - 2 * self.roller.reduce
        # 计算滚柱牙根高
        self.roller.c = (0.5 - self.roller.cutter_bottom) * self.high_thread
        # 计算丝杠中径牙厚
        self.screw.b = 0.5 * self.screw.pitch - 2 * self.screw.reduce
        # 计算丝杠底径牙厚
        self.screw.a = self.screw.pitch * (1 - self.screw.cutter_bottom) - 2 * self.screw.reduce
        # 计算丝杠牙根高
        self.screw.c = (0.5 - self.screw.cutter_bottom) * self.high_thread

        # 计算螺母啮合柔度
        flexible_nut_1 = (
            (1 - self.nut.nu ** 2) * 
            (3 / (4 * self.nut.E)) * 
            (
                (1 - (2 - self.nut.b / self.nut.a) ** 2 + 2 * np.log(self.nut.a / self.nut.b)) / 
                (np.tan(0.5 * self.screw.angle) ** 3) - 
                4 * (self.nut.c / self.nut.a) ** 2 * np.tan(0.5 * self.screw.angle)
            )
        )

        flexible_nut_2 = (
            (1 + self.nut.nu) * 
            (6 / (5 * self.nut.E)) * 
            (1 / np.tan(0.5 * self.screw.angle)) * 
            np.log(self.nut.a / self.nut.b)
        )

        flexible_nut_3 = (
            (1 - self.nut.nu ** 2) * 
            (12 * self.nut.c / (np.pi * self.nut.E * self.nut.a ** 2)) * 
            (self.nut.c - 0.5 * self.nut.b * np.tan(0.5 * self.screw.angle))
        )

        flexible_nut_4 = (
            (1 - self.nut.nu ** 2) * 
            (2 / (np.pi * self.nut.E)) * 
            (
                (self.screw.pitch / self.nut.a) * 
                np.log((self.screw.pitch + 0.5 * self.nut.a) / (self.screw.pitch - 0.5 * self.nut.a)) + 
                0.5 * np.log(4 * self.screw.pitch ** 2 / self.nut.a ** 2 - 1)
            )
        )

        flexible_nut_5 = (
            (
                (self.nut.diameter_outer ** 2 + self.nut.D_pitch ** 2) / 
                (self.nut.diameter_outer ** 2 - self.nut.D_pitch ** 2) + 
                self.nut.nu
            ) * 
            0.5 * np.tan(0.5 * self.screw.angle) ** 2 * 
            (self.nut.D_pitch / self.screw.pitch) / 
            self.nut.E
        )

        flexible_nut = flexible_nut_1 + flexible_nut_2 + flexible_nut_3 + flexible_nut_4 + flexible_nut_5

        # 螺杆啮合柔度计算
        flexible_screw_1 = (
            (1 - self.screw.nu ** 2) * (3 / (4 * self.screw.E)) * 
            (
                (1 - (2 - self.screw.b / self.screw.a) ** 2 + 2 * np.log(self.screw.a / self.screw.b)) 
                / (np.tan(0.5 * self.screw.angle) ** 3) 
                - 4 * (self.screw.c / self.screw.a) ** 2 * np.tan(0.5 * self.screw.angle)
            )
        )

        flexible_screw_2 = (
            (1 + self.screw.nu) * (6 / (5 * self.screw.E)) * 
            (1 / np.tan(0.5 * self.screw.angle)) * 
            np.log(self.screw.a / self.screw.b)
        )

        flexible_screw_3 = (
            (1 - self.screw.nu ** 2) * 
            (12 * self.screw.c / (np.pi * self.screw.E * self.screw.a ** 2)) * 
            (self.screw.c - 0.5 * self.screw.b * np.tan(0.5 * self.screw.angle))
        )

        flexible_screw_4 = (
            (1 - self.screw.nu ** 2) * (2 / (np.pi * self.screw.E)) * 
            (
                (self.screw.pitch / self.screw.a) * 
                np.log((self.screw.pitch + 0.5 * self.screw.a) / (self.screw.pitch - 0.5 * self.screw.a)) 
                + 0.5 * np.log(4 * self.screw.pitch ** 2 / self.screw.a ** 2 - 1)
            )
        )

        flexible_screw_5 = (
            (1 - self.screw.nu) * 0.5 * 
            np.tan(0.5 * self.screw.angle) ** 2 * 
            (self.screw.D_pitch / self.screw.pitch) / self.screw.E
        )

        flexible_screw = flexible_screw_1 + flexible_screw_2 + flexible_screw_3 + flexible_screw_4 + flexible_screw_5

        # 滚柱啮合柔度计算
        flexible_roller_1 = (
            (1 - self.roller.nu ** 2) * (3 / (4 * self.roller.E)) * 
            (
                (1 - (2 - self.roller.b / self.roller.a) ** 2 + 2 * np.log(self.roller.a / self.roller.b)) 
                / (np.tan(0.5 * self.screw.angle) ** 3)  # 注意此处共用螺杆角度
                - 4 * (self.roller.c / self.roller.a) ** 2 * np.tan(0.5 * self.screw.angle)
            )
        )

        flexible_roller_2 = (
            (1 + self.roller.nu) * (6 / (5 * self.roller.E)) * 
            (1 / np.tan(0.5 * self.screw.angle)) * 
            np.log(self.roller.a / self.roller.b)
        )

        flexible_roller_3 = (
            (1 - self.roller.nu ** 2) * 
            (12 * self.roller.c / (np.pi * self.roller.E * self.roller.a ** 2)) * 
            (self.roller.c - 0.5 * self.roller.b * np.tan(0.5 * self.screw.angle))
        )

        flexible_roller_4 = (
            (1 - self.roller.nu ** 2) * (2 / (np.pi * self.roller.E)) * 
            (
                (self.screw.pitch / self.roller.a) *  # 注意pitch来自螺杆参数
                np.log((self.screw.pitch + 0.5 * self.roller.a) / (self.screw.pitch - 0.5 * self.roller.a)) 
                + 0.5 * np.log(4 * self.screw.pitch ** 2 / self.roller.a ** 2 - 1)
            )
        )

        flexible_roller_5 = (
            (1 - self.roller.nu) * 0.5 * 
            np.tan(0.5 * self.screw.angle) ** 2 * 
            (self.roller.D_pitch / self.screw.pitch) / self.roller.E
        )

        flexible_roller = flexible_roller_1 + flexible_roller_2 + flexible_roller_3 + flexible_roller_4 + flexible_roller_5
        
        # 计算内螺纹的啮合柔度
        self.flexible_inner = np.abs(flexible_nut + flexible_roller) / 1

        # 计算外螺纹的啮合柔度
        self.flexible_outer = np.abs(flexible_screw + flexible_roller) / 1

    def solver(self):
        if self.type_load == 1:
            # 初始值
            F0 = self.nut.force * np.ones([2 * self.number, self.number_roller])
            # 螺母侧零工作侧隙
            nut_temp = np.zeros([self.number, self.number_roller])
            # 丝杠侧零工作侧隙
            screw_temp = np.zeros([self.number, self.number_roller])
            # 螺母侧工作侧隙极大值向量
            max_nut = np.zeros([self.number, self.number_roller])
            # 丝杠侧工作侧隙极大值向量
            max_screw = np.zeros([self.number, self.number_roller])
            # 滚柱载荷非配系数
            share = (1 / self.number_roller) * np.ones([self.number_roller, 1])
            # 调用fsolve求解零工作侧隙的法向载荷
            Fn_refer = fsolve(lambda Fn: load_roller_screw_1(Fn, self.force_axial, self.pitch_screw, self.number_roller, self.nut.ratio, self.screw.ratio,
                                                 self.nut.E, self.screw.E, self.roller.E, self.nut.area, self.screw.area, self.roller.area, self.C_delta_inner, self.C_delta_outer,
                                                 self.E0_inner, self.E0_outer, self.p_inner_sum, self.p_outer_sum, self.flexible_inner, self.flexible_outer,
                                                 nut_temp, screw_temp, share), F0, xtol=1e-6)
            Fn_refer = Fn_refer.reshape(62, 5)
            
            # 计算考虑实际工作侧隙的各接触点法向载荷
            if (np.sum(self.nut.modify) + np.sum(self.screw.modify)) > 0:
                # 计算螺母侧/丝杠侧工作侧隙极值
                for kk in range (self.number_roller):
                    for ii in range (1, self.number):
                        Fn_temp = Fn_temp.copy()
                        Fn_temp[ii-1, kk] = 0
                        Fn_temp[ii-1+self.number, kk] = 0

                        max_nut[ii-1, kk] = (
                            np.sum(np.maximum(0, Fn_temp[ii: self.number, 0: self.number_roller])) * 
                            self.nut.ratio * self.screw.pitch / (self.nut.area * self.nut.E) - 
                            self.C_delta_inner * self.E0_inner ** 2 * self.p_inner_sum ** (1 / 3) * 
                            (
                                np.maximum(0, Fn_temp[ii-1, kk]) ** (2 / 3) - 
                                np.max(0, Fn_temp[ii, kk] ** (2 / 3)) ** (2 / 3)
                            ) / 
                            self.nut.ratio - self.flexible_inner * 
                            (np.maximum(0, Fn_temp[ii-1, kk]) - np.maximum(0, Fn_temp[ii, kk])) + 
                            (
                                np.sum(np.maximum(0, Fn_temp[ii, self.number, kk])) * self.nut.ratio - 
                             np.sum(np.maximum(0, Fn_temp[ii+self.number:2*self.number, kk])) * self.screw.ratio
                            ) * 
                             self.screw.pitch / (self.roller.area * self.roller.E)
                        )

                        max_screw[ii-1, kk] = (
                            np.sum(np.maximum(0, Fn_temp[ii+self.number:2*self.number, 0:self.number_roller])) * 
                            self.screw.ratio * self.screw.pitch / (self.screw.area * self.screw.E) - 
                            self.C_delta_outer * self.E0_outer ** 2 * self.p_outer_sum * (1 / 3) * 
                            (
                                np.maximum(0, Fn_temp[ii-1+self.number, kk]) ** (2 / 3) - 
                                np.maximum(0, Fn_temp[ii+self.number, kk]) ** (2 / 3)
                            ) / 
                            self.screw.ratio - self.flexible_outer * 
                            (
                                np.maximum(0, Fn_temp[ii-1+self.number, kk]) - 
                                np.maximum(0, Fn_temp[ii+self.number, kk])
                            ) - 
                            (
                                np.sum(np.maximum(0, Fn_temp[ii:self.number, kk])) * self.nut.ratio - 
                                np.sum(np.maximum(0, Fn_temp[ii+self.number:2*self.number, kk]))
                            ) * self.screw.pitch / (self.roller.area * self.roller.E)
                        )
                    max_nut[ii, kk] = max_nut[ii-1, kk].copy()
                    max_screw[ii, kk] = max_screw[ii-1, kk].copy()   
                max_nut = 0.9 * max_nut
                max_screw = 0.9 * max_screw
                # 调整螺母侧/丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)
                self.nut.modify = np.minimum(self.nut.modify, max_nut)
                self.screw.modify = np.minimum(self.screw.modify, max_screw)
                # 求解考虑工作侧隙时行星滚柱丝杠各接触点的受载法向载荷
                Fn, info, ier, msg = fsolve(lambda Fn: load_roller_screw_1(Fn, self.force_axial, self.pitch_screw, self.number_roller, 
                                                           self.nut.ratio, self.screw.ratio, self.nut.E, self.screw.E, 
                                                           self.roller.E, self.nut.area, self.screw.area, self.roller.area, 
                                                           self.C_delta_inner, self.C_delta_outer, self.E0_inner, self.E0_outer, 
                                                           self.p_inner_sum, self.p_outer_sum, self.flexible_inner, self.flexible_outer, 
                                                           modify_nut, modify_screw, share), 
                            F0, xtol=1e-6, full_output=True)
                Fn = Fn.reshpe(62, 5)
                fval = info['fvec']

                

                # 计算 `fval` 是否超出阈值
                if (np.sum(np.abs(fval[1:self.number, 0:self.number_roller])) > 0.01 or
                    np.sum(np.abs(fval[1+self.number:2*self.number, 0:self.number_roller])) > 0.01):

                    # 创建消息框窗口
                    root = tk.Tk()
                    root.withdraw()  # 隐藏主窗口
                    messagebox.showwarning("警告", "行星滚柱丝杠法向载荷计算存在不收敛")

                    # 显示自定义图标（替换 `飞机.jpg`）
                    try:
                        img = Image.open("飞机.jpg")
                        img.show()  # 这会用默认图片查看器打开图片
                    except Exception as e:
                        print("无法加载图片:", e)
            else:
                Fn = Fn_refer.copy()
            # 调整行星滚柱丝杠各接触点的受载法向载荷
            Fn = np.maximum(0, Fn)
            # 记录螺母的接触点法向载荷的向量
            Fn_nut = np.zeros((self.number, 1 + self.number_roller))
            Fn_nut[:, 0] = np.arange(1, self.number + 1)
            Fn_nut[:, 1:] = Fn[:self.number, :self.number_roller]
            # 记录螺母的接触点法向载荷不均载系数的向量
            distribution_nut = np.zeros([self.number, 1 + self.number_roller])
            distribution_nut[:, 0] = np.arange(1, self.number + 1)
            distribution_nut[:, 1:] = Fn_nut[:, 1:] / (self.force_axial / self.number_load / self.nut.ratio)

            # 丝杠的接触点法向载荷的向量
            Fn_screw = np.zeros((self.number, 1 + self.number_roller))
            Fn_screw[:, 0] = np.arange(1, self.number + 1)  
            Fn_screw[:, 1:] = Fn[self.number:2*self.number, :self.number_roller]  

            # 计算丝杠接触点法向载荷不均载系数
            distribution_screw = np.zeros((self.number, 1 + self.number_roller))
            distribution_screw[:, 0] = Fn_screw[:, 0]  
            distribution_screw[:, 1:] = Fn_screw[:, 1:] / (self.force_axial / self.number_load / self.screw.ratio)
            
        if type_load == 2:  # 行星滚柱丝杠受载方式，2 表示异侧受载
            # 设置初始值
            F0 = self.nut.force * np.ones((2 * self.number, self.number_roller))
            
            # 初始化变量
            nut_temp = np.zeros((self.number, self.number_roller))
            screw_temp = np.zeros((self.number, self.number_roller))
            max_nut = np.zeros((self.number, self.number_roller))
            max_screw = np.zeros((self.number, self.number_roller))
            share = (1 / self.number_roller) * np.ones(self.number_roller)
            
            # 求解零工作侧隙的法向载荷
            Fn_refer = fsolve(lambda Fn: load_roller_screw_2(Fn, self.force_axial, self.pitch_screw, self.number_roller, self.nut.ratio, self.screw.ratio,
                                                 self.nut.E, self.screw.E, self.roller.E, self.nut.area, self.screw.area, self.roller.area, self.C_delta_inner, self.C_delta_outer,
                                                 self.E0_inner, self.E0_outer, self.p_inner_sum, self.p_outer_sum, self.flexible_inner, self.flexible_outer,
                                                 nut_temp, screw_temp, share), F0, xtol=1e-6)
            Fn_refer = Fn_refer.reshape(62, 5)
            
            # 计算考虑实际工作侧隙的法向载荷
            if np.sum(self.nut.modify) + np.sum(self.screw.modify) > 0:
                for kk in range(self.number_roller):
                    for ii in range(1, self.number):
                        Fn_temp = Fn_refer.copy()
                        Fn_temp[ii - 1, kk] = 0
                        Fn_temp[ii + self.number, kk] = 0
                        
                        max_nut[ii - 1, kk] = (np.sum(np.maximum(0, Fn_temp[ii:self.number, :])) * self.nut.ratio * self.screw.pitch / (self.nut.area * self.nut.E)
                                            - self.C_delta_inner * self.E0_inner ** 2 * self.p_inner_sum ** (1 / 3) *
                                                (np.maximum(0, Fn_temp[ii - 1, kk])**(2/3) - np.maximum(0, Fn_temp[ii, kk])**(2/3)) / self.nut.ratio
                                            - self.flexible_inner * (np.maximum(0, Fn_temp[ii - 1, kk]) - np.maximum(0, Fn_temp[ii, kk]))
                                            + (np.sum(np.maximum(0, Fn_temp[ii:self.number, kk])) * self.nut.ratio -
                                                np.sum(np.maximum(0, Fn_temp[ii + self.number:2 * self.number, kk])) * self.screw.ratio)
                                                * self.screw.pitch / (self.roller.area * self.roller.E))
                        
                        max_screw[ii, kk] = (np.sum(np.maximum(0, Fn_temp[self.number:ii - 1 + self.number, :])) * self.screw.ratio * self.screw.pitch / (self.screw.area * self.screw.E)
                                            - self.C_delta_outer * self.E0_outer**2 * self.p_outer_sum ** (1 / 3) *
                                                (np.maximum(0, Fn_temp[ii + self.number, kk])**(2/3) - np.maximum(0, Fn_temp[ii - 1 + self.number, kk])**(2/3)) / self.screw.ratio
                                            - self.flexible_outer * (np.maximum(0, Fn_temp[ii + self.number, kk]) - np.maximum(0, Fn_temp[ii - 1 + self.number, kk]))
                                            + (np.sum(np.maximum(0, Fn_temp[ii:self.number, kk])) * self.nut.ratio -
                                                np.sum(np.maximum(0, Fn_temp[ii + self.number:2 * self.number, kk])) * self.screw.ratio)
                                                * self.screw.pitch / (self.roller.area * self.roller.E))
                    max_nut[ii, kk] = max_nut[ii - 1, kk]
                    max_screw[0, kk] = max_screw[1, kk]
                
                max_nut *= 0.90
                max_screw *= 0.90
                modify_nut = np.minimum(modify_nut, max_nut)
                modify_screw = np.minimum(modify_screw, max_screw)
                
                Fn, fval, _, _ = fsolve(lambda Fn: load_roller_screw_2(Fn, self.force_axial, self.pitch_screw, self.number_roller, self.nut.ratio, self.screw.ratio,
                                                 self.nut.E, self.screw.E, self.roller.E, self.nut.area, self.screw.area, self.roller.area, self.C_delta_inner, self.C_delta_outer,
                                                 self.E0_inner, self.E0_outer, self.p_inner_sum, self.p_outer_sum, self.flexible_inner, self.flexible_outer,
                                                 nut_temp, screw_temp, share),
                                        F0, full_output=True)
                
                if np.any(np.abs(fval[1:self.number, :]) > 0.01) or np.any(np.abs(fval[self.number + 1:2 * self.number, :]) > 0.01):
                    print("警告: 行星滚柱丝杠法向载荷计算可能未收敛")
            else:
                Fn = Fn_refer
            
            Fn = np.maximum(0, Fn)
            
            Fn_nut = np.column_stack((np.arange(1, self.number + 1), Fn[:self.number, :]))
            distribution_nut = np.column_stack((np.arange(1, self.number + 1), Fn_nut[:, 1:] / (self.force_axial / self.number_load / self.nut.ratio)))
            
            Fn_screw = np.column_stack((np.arange(1, self.number + 1), Fn[self.number:2 * self.number, :]))
            distribution_screw = np.column_stack((np.arange(1, self.number + 1), Fn_screw[:, 1:] / (self.force_axial / self.number_load / self.screw.ratio)))
        
        self.screw.Fn = Fn_screw
        self.nut.Fn = Fn_nut
        self.distribution_nut = distribution_nut
        self.distribution_screw = distribution_screw

    def compute_half_width(self):
        # 外啮合接触长半轴
        self.half_width_a_outer = np.zeros((self.number, self.number_roller + 1))
        self.half_width_b_outer = np.zeros((self.number, self.number_roller + 1))
        self.half_width_a_outer[:, 0] = np.arange(1, self.number + 1)
        self.half_width_a_outer[:, 1:] = self.C_a_outer * self.E0_outer * (self.screw.Fn[:, 1:] / self.p_outer_sum) ** (1 / 3)
        self.half_width_b_outer[:, 0] = np.arange(1, self.number + 1)
        self.half_width_b_outer[:, 1:] = self.C_b_outer * self.E0_outer * (self.screw.Fn[:, 1:] / self.p_outer_sum) ** (1 / 3)
        # 内啮合接触短半轴
        self.half_width_a_inner = np.zeros((self.number, self.number_roller + 1))
        self.half_width_b_inner = np.zeros((self.number, self.number_roller + 1))
        self.half_width_a_inner[:, 0] = np.arange(1, self.number + 1)
        self.half_width_a_inner[:, 1:] = self.C_a_inner * self.E0_inner * (self.nut.Fn[:, 1:] / self.p_inner_sum) ** (1 / 3)
        self.half_width_b_inner[:, 0] = np.arange(1, self.number + 1)
        self.half_width_b_inner[:, 1:] = self.C_b_inner * self.E0_inner * (self.nut.Fn[:, 1:] / self.p_inner_sum) ** (1 / 3)


    def compute_stress(self):
        # 外啮合接触应力
        self.stress_outer = np.zeros((self.number, self.number_roller + 1))
        self.stress_outer[:, 0] = np.arange(1, self.number + 1)
        self.stress_outer[:, 1:] = 1.5 * self.screw.Fn[:, 1:] / (np.pi * self.half_width_a_outer[:, 1:] * self.half_width_b_outer[:, 1:])
        # 内啮合接触应力
        self.stress_inner = np.zeros((self.number, self.number_roller + 1))
        self.stress_inner[:, 0] = np.arange(1, self.number + 1)
        self.stress_inner[:, 1:] = 1.5 * self.nut.Fn[:, 1:] / (np.pi * self.half_width_a_inner[:, 1:] * self.half_width_b_inner[:, 1:])


    def computer_delta(self):
        # 外啮合变形量
        self.delta_outer = np.zeros((self.number, self.number_roller + 1))
        self.delta_outer[:, 0] = np.arange(1, self.number + 1)
        self.delta_outer[:, 1:] = self.C_delta_outer * (self.E0_outer ** 2) * (self.p_outer_sum * (self.screw.Fn[:, 1:] ** 2)) ** (1 / 3)
        # 内啮合变形量
        self.delta_inner = np.zeros((self.number, self.number_roller + 1))
        self.delta_inner[:, 0] = np.arange(1, self.number + 1)
        self.delta_inner[:, 1:] = self.C_delta_inner * (self.E0_inner ** 2) * (self.p_inner_sum * (self.nut.Fn[:, 1:] ** 2)) ** (1 / 3)

    def compute_s_load(self):
        # 螺母载荷分布均方值
        self.nut.s_load = (
            np.sum((self.nut.Fn[:, 1:] - np.sum(self.nut.Fn[:, 1:], axis=0) / self.number) ** 2, axis=0) / self.number
        ) ** (0.5)

        # 丝杠载荷分布均方值
        self.screw.s_load = (
            np.sum((self.screw.Fn[:, 1:] - np.sum(self.screw.Fn[:, 1:], axis=0) / self.number) ** 2, axis=0) / self.number
        ) ** (0.5)

    def potential_roller_screw(self):
        self.roller_screw_segment_produce()
        self.roller_screw_stiffness_produce()
        # 螺母轴向载荷向量
        Axial_nut = np.zeros((6 * (2 + self.number), 1))
        for ii in range(self.number):
            Axial_nut[6*(ii+1)+3-1, 0] = -1 * np.sum(self.nut.Fn[ii, 1:1+self.number_roller]) * self.nut.ratio
        Axial_nut[2, 0] = np.sum(self.nut.Fn[:, 1:1+self.number_roller]) * self.nut.ratio
        # 螺母刚度矩阵
        id_stiffness_nut = self.roller_screw_stiffness[:, 0] == (int(self._order_roller_screw) + 0.0001)
        stiffness_nut = self.roller_screw_stiffness[id_stiffness_nut, 1:1+6*(2+self.number)]
        # 螺母柔度矩阵
        flexible_nut = np.linalg.inv(stiffness_nut)
        # 螺母拉伸/压缩势能
        potential_nut = 0.5 * Axial_nut.T @ flexible_nut @ Axial_nut

        # 丝杠轴向载荷向量
        Axial_screw = np.zeros((6*(2+self.number), 1))
        for ii in range(self.number):
            Axial_screw[6*(ii+1)+3-1, 0] = np.sum(self.screw.Fn[ii, 1:1+self.number_roller]) * self.screw.ratio
        if self.type_load == 1:
            Axial_screw[2, 0] = -1 * np.sum(self.screw.Fn[:, 1:1+self.number_roller]) * self.screw.ratio
        elif self.type_load == 2:
            Axial_screw[6*(1+self.number)+2, 0] = -1 * np.sum(self.screw.Fn[:, 1:1+self.number_roller]) * self.screw.ratio
        id_stiffness_screw =  self.roller_screw_stiffness[:, 0] == (int(self._order_roller_screw) + 0.0002)
        stiffness_screw = self.roller_screw_stiffness[id_stiffness_screw, 1:1+6*(2+self.number)]
        flexible_screw = np.linalg.inv(stiffness_screw)
        potential_screw = 0.5 * Axial_screw.T @ flexible_screw @ Axial_screw

        # 滚柱
        potential_roller = 0
        id_stiffness_roller =  self.roller_screw_stiffness[:, 0] == (int(self._order_roller_screw) + 0.0004)
        stiffness_roller = self.roller_screw_stiffness[id_stiffness_roller, 1:1+6*(2+self.number)]
        flexible_roller = np.linalg.inv(stiffness_roller)
        for kk in range(self.number_roller):
            Axial_roller = np.zeros((6*(2+self.number), 1))
            for ii in range(self.number):
                Axial_roller[6*(ii+1)+2, 0] = self.nut.Fn[ii, kk+1] * self.nut.ratio - self.screw.Fn[ii, 1+kk] * self.screw.ratio
            potential_roller = potential_roller + 0.5 * Axial_roller.T @ flexible_roller @ Axial_roller
        # 螺母/丝杠/滚柱轴向拉伸/压缩势能总和
        self.potential_axial = potential_nut + potential_screw + potential_roller

        potential_bend_inner = np.sum(
            0.5 * self.nut.Fn[:self.number, 1:1+self.number_roller] * 
            self.flexible_inner * self.nut.Fn[:self.number, 1:1+self.number_roller]
        )
        potential_bend_outer = np.sum(
            0.5 * self.screw.Fn[:self.number, 1:1+self.number_roller] * 
            self.flexible_outer * self.screw.Fn[:self.number, 1:1+self.number_roller]
        )
        # 螺纹弯曲势能总和
        self.potential_bend = potential_bend_inner + potential_bend_outer

        
        potential_compress_inner = np.sum(0.25 * np.pi * self.B_inner * self.half_width_a_inner[:self.number, 1:1+self.number_roller] * (self.half_width_b_inner[:self.number, 1:1+self.number_roller]**2) * self.stress_inner[:self.number, 1:1+self.number_roller])
        potential_compress_outer = np.sum(0.25 * np.pi * self.B_outer * self.half_width_a_outer[:self.number, 1:1+self.number_roller] * (self.half_width_b_outer[:self.number, 1:1+self.number_roller]**2) * self.stress_outer[:self.number, 1:1+self.number_roller])
        # 螺纹压缩势能总和
        self.potential_compress = potential_compress_inner + potential_compress_outer
        
        # 行星滚柱丝杠总的势能
        self.potential_total = self.potential_axial + self.potential_bend + self.potential_compress

    def roller_screw_segment_produce(self):
        self.roller_screw_segment = np.zeros(((self.number + 1) * 3, 16))
        self.computer_distance()
        self.shaft_segment_nut = np.zeros((self.number + 1, 16))
        self.shaft_segment_roller = np.zeros((self.number + 1, 16))
        self.shaft_segment_screw = np.zeros((self.number + 1, 16))
        # 生成螺母、丝杠、滚柱轮缘轴的网格段矩阵
        for kk in range(self.number + 1):
            if kk == 0:
                self.shaft_segment_nut[kk, 0] = self.nut.shaft_self
                self.shaft_segment_nut[kk, 1] = kk + 1
                self.shaft_segment_nut[kk, 2] = self.distance_nut_left
                self.shaft_segment_nut[kk, 3] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 4] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 5] = self.nut.E
                self.shaft_segment_nut[kk, 6] = self.nut.nu
                self.shaft_segment_nut[kk, 7] = 0.886
                self.shaft_segment_nut[kk, 8] = self.distance_nut_left
                self.shaft_segment_nut[kk, 9] = self.nut.density
                self.shaft_segment_nut[kk, 10] = 1
                self.shaft_segment_nut[kk, 11] = 0
                self.shaft_segment_nut[kk, 12] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 13] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 14] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 15] = self.nut.D_bottom

                self.shaft_segment_screw[kk, 0] = self.screw.shaft_self
                self.shaft_segment_screw[kk, 1] = kk + 1
                self.shaft_segment_screw[kk, 2] = self.distance_screw_left
                self.shaft_segment_screw[kk, 3] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 4] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 5] = self.screw.E
                self.shaft_segment_screw[kk, 6] = self.screw.nu
                self.shaft_segment_screw[kk, 7] = 0.886
                self.shaft_segment_screw[kk, 8] = self.distance_screw_left
                self.shaft_segment_screw[kk, 9] = self.screw.density
                self.shaft_segment_screw[kk, 10] = 1
                self.shaft_segment_screw[kk, 11] = 0
                self.shaft_segment_screw[kk, 12] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 13] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 14] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 15] = self.screw.diameter_inner

                self.shaft_segment_roller[kk, 0] = self.roller.shaft_self
                self.shaft_segment_roller[kk, 1] = kk + 1
                self.shaft_segment_roller[kk, 2] = self.distance_roller_left
                self.shaft_segment_roller[kk, 3] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 4] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 5] = self.roller.E
                self.shaft_segment_roller[kk, 6] = self.roller.nu
                self.shaft_segment_roller[kk, 7] = 0.886
                self.shaft_segment_roller[kk, 8] = self.distance_roller_left
                self.shaft_segment_roller[kk, 9] = self.roller.density
                self.shaft_segment_roller[kk, 10] = 1
                self.shaft_segment_roller[kk, 11] = 0
                self.shaft_segment_roller[kk, 12] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 13] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 14] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 15] = self.roller.diameter_inner

            elif kk == self.number:
                self.shaft_segment_nut[kk, 0] = self.nut.shaft_self
                self.shaft_segment_nut[kk, 1] = kk + 1
                self.shaft_segment_nut[kk, 2] = self.nut.length_shaft - self.distance_nut_right
                self.shaft_segment_nut[kk, 3] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 4] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 5] = self.nut.E
                self.shaft_segment_nut[kk, 6] = self.nut.nu
                self.shaft_segment_nut[kk, 7] = 0.886
                self.shaft_segment_nut[kk, 8] = self.nut.length_shaft
                self.shaft_segment_nut[kk, 9] = self.nut.density
                self.shaft_segment_nut[kk, 10] = 1
                self.shaft_segment_nut[kk, 11] = 0
                self.shaft_segment_nut[kk, 12] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 13] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 14] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 15] = self.nut.D_bottom

                self.shaft_segment_screw[kk, 0] = self.screw.shaft_self
                self.shaft_segment_screw[kk, 1] = kk + 1
                self.shaft_segment_screw[kk, 2] = self.screw.length_shaft - self.distance_screw_right
                self.shaft_segment_screw[kk, 3] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 4] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 5] = self.screw.E
                self.shaft_segment_screw[kk, 6] = self.screw.nu
                self.shaft_segment_screw[kk, 7] = 0.886
                self.shaft_segment_screw[kk, 8] = self.screw.length_shaft
                self.shaft_segment_screw[kk, 9] = self.screw.density
                self.shaft_segment_screw[kk, 10] = 1
                self.shaft_segment_screw[kk, 11] = 0
                self.shaft_segment_screw[kk, 12] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 13] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 14] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 15] = self.screw.diameter_inner

                self.shaft_segment_roller[kk, 0] = self.roller.shaft_self
                self.shaft_segment_roller[kk, 1] = kk + 1
                self.shaft_segment_roller[kk, 2] = self.roller.length_shaft - self.distance_roller_right
                self.shaft_segment_roller[kk, 3] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 4] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 5] = self.roller.E
                self.shaft_segment_roller[kk, 6] = self.roller.nu
                self.shaft_segment_roller[kk, 7] = 0.886
                self.shaft_segment_roller[kk, 8] = self.roller.length_shaft
                self.shaft_segment_roller[kk, 9] = self.roller.density
                self.shaft_segment_roller[kk, 10] = 1
                self.shaft_segment_roller[kk, 11] = 0
                self.shaft_segment_roller[kk, 12] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 13] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 14] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 15] = self.roller.diameter_inner

            else:
                self.shaft_segment_nut[kk, 0] = self.nut.shaft_self
                self.shaft_segment_nut[kk, 1] = kk + 1
                self.shaft_segment_nut[kk, 2] = self.screw.pitch
                self.shaft_segment_nut[kk, 3] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 4] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 5] = self.nut.E
                self.shaft_segment_nut[kk, 6] = self.nut.nu
                self.shaft_segment_nut[kk, 7] = 0.886
                self.shaft_segment_nut[kk, 8] = self.distance_nut_left + kk * self.screw.pitch
                self.shaft_segment_nut[kk, 9] = self.nut.density
                self.shaft_segment_nut[kk, 10] = 1
                self.shaft_segment_nut[kk, 11] = 0
                self.shaft_segment_nut[kk, 12] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 13] = self.nut.diameter_outer
                self.shaft_segment_nut[kk, 14] = self.nut.D_bottom
                self.shaft_segment_nut[kk, 15] = self.nut.D_bottom

                self.shaft_segment_screw[kk, 0] = self.screw.shaft_self
                self.shaft_segment_screw[kk, 1] = kk + 1
                self.shaft_segment_screw[kk, 2] = self.screw.pitch
                self.shaft_segment_screw[kk, 3] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 4] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 5] = self.screw.E
                self.shaft_segment_screw[kk, 6] = self.screw.nu
                self.shaft_segment_screw[kk, 7] = 0.886
                self.shaft_segment_screw[kk, 8] = self.distance_screw_left + kk * self.screw.pitch
                self.shaft_segment_screw[kk, 9] = self.screw.density
                self.shaft_segment_screw[kk, 10] = 1
                self.shaft_segment_screw[kk, 11] = 0
                self.shaft_segment_screw[kk, 12] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 13] = self.screw.D_bottom
                self.shaft_segment_screw[kk, 14] = self.screw.diameter_inner
                self.shaft_segment_screw[kk, 15] = self.screw.diameter_inner

                self.shaft_segment_roller[kk, 0] = self.roller.shaft_self
                self.shaft_segment_roller[kk, 1] = kk + 1
                self.shaft_segment_roller[kk, 2] = self.screw.pitch
                self.shaft_segment_roller[kk, 3] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 4] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 5] = self.roller.E
                self.shaft_segment_roller[kk, 6] = self.roller.nu
                self.shaft_segment_roller[kk, 7] = 0.886
                self.shaft_segment_roller[kk, 8] = self.distance_roller_left + kk * self.screw.pitch
                self.shaft_segment_roller[kk, 9] = self.roller.density
                self.shaft_segment_roller[kk, 10] = 1
                self.shaft_segment_roller[kk, 11] = 0
                self.shaft_segment_roller[kk, 12] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 13] = self.roller.D_bottom
                self.shaft_segment_roller[kk, 14] = self.roller.diameter_inner
                self.shaft_segment_roller[kk, 15] = self.roller.diameter_inner
        
        self.roller_screw_segment = np.concatenate((self.shaft_segment_nut, self.shaft_segment_screw, self.shaft_segment_roller))
        
        # 更新轴的刚度矩阵
        for ii in range((self.number + 1) * 3):
            poisson = self.roller_screw_segment[ii, 7-1]
            diameter_outer = self.roller_screw_segment[ii, 4-1]
            diameter_inner = self.roller_screw_segment[ii, 5-1]
            diameter_ratio = diameter_inner / diameter_outer
            a_factor = 6 * (1 + poisson) * (1 + diameter_ratio ** 2) ** 2 / ((7 + 6 * poisson) * (1 + diameter_ratio ** 2) ** 2 + (20 + 12 * poisson) * diameter_ratio **2)
            self.roller_screw_segment[ii, 8-1] = a_factor

    
    def computer_distance(self):
        # 丝杠螺纹左端相对于丝杠轴左端的位置
        self.thread_left_s = self.screw.delta_thread
        # 滚柱螺纹左端相对于丝杠轴左端的位置
        self.thread_left_r = self.roller.delta_thread + self.delta_r_s
        # 滚柱螺纹左端相对于滚柱轴左端的位置
        self.thread_left_n = self.nut.delta_thread + self.delta_n_s
        # 螺母有效螺纹左端距螺母轴左端的距离
        self.distance_nut_left = np.max(np.array([self.thread_left_n, self.thread_left_s, self.thread_left_r])) - self.delta_n_s
        # 螺母有效螺纹右端距螺母轴左端的距离
        self.distance_nut_right = self.distance_nut_left + (self.number - 1) * self.screw.pitch
        # 丝杠有效螺纹左端距丝杠轴左端的距离
        self.distance_screw_left = np.max(np.array([self.thread_left_n, self.thread_left_s, self.thread_left_r]))
        # 丝杠有效螺纹右端距丝杠轴左端的距离
        self.distance_screw_right = self.distance_screw_left + (self.number - 1) * self.screw.pitch
        # 滚柱有效螺纹左端距滚柱轴左端的距离
        self.distance_roller_left = np.max(np.array([self.thread_left_n, self.thread_left_s, self.thread_left_r])) - self.delta_r_s
        # 滚柱有效螺纹右端距滚柱轴左端的距离
        self.distance_roller_right = self.distance_roller_left + (self.number - 1) * self.screw.pitch


    def roller_screw_stiffness_produce(self):
        # 行星滚柱丝杠的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵
        self.roller_screw_stiffness = np.zeros((3 * 6 * (self.number + 2), 1 + 6 * (self.number + 2)))
        dd = 0
        self._order_roller_screw
        for mm in range(1, 4):
            if mm == 3:
                order_shaft = int(self._order_roller_screw) + 0.0001 * (mm + 1)
            else:
                order_shaft = int(self._order_roller_screw) + 0.0001 * mm

            id_shaft_segment = self.roller_screw_segment[:, 0] == order_shaft
            shaft_segment_single = self.roller_screw_segment[id_shaft_segment, 1:10]
            shaft_segment_single = shaft_segment_single[shaft_segment_single[:, 0].argsort()]
            shaft_segment = np.sum(id_shaft_segment) + 1
            shaft_single_stiffness = np.zeros((6 * shaft_segment, 6 * shaft_segment))
            shaft_single_stiffness_local = np.zeros((6 * shaft_segment, 6 * shaft_segment))
            # 计算各轴段的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵
            for k in range(shaft_segment):
                if k == 0:
                    # 弯曲振动刚度
                    I_section = np.pi * (shaft_segment_single[k ,2] ** 4 - shaft_segment_single[k, 3] ** 4) / 64
                    L_shaft = shaft_segment_single[k, 1]
                    E = shaft_segment_single[k, 4]
                    possion = shaft_segment_single[k, 5]  
                    a_factor = shaft_segment_single[k, 6]
                    G = E / (2 * (1 + possion))
                    area = (np.pi / 4) * (shaft_segment_single[k, 2]**2 - shaft_segment_single[k, 3]**2)
                    if a_factor == 0:
                        f_sheer = 0
                    else:
                        f_sheer = 6 * E * I_section / (a_factor * G * area * L_shaft**2)
                    beta_1 = 12 * E * I_section / (L_shaft**3 * (1 + 2 * f_sheer))
                    beta_2 = 0.5 * L_shaft * beta_1
                    beta_3 = L_shaft**2 * (1 - f_sheer) * beta_1 / 6
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-6] = beta_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-5] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)] = -beta_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)+1] = beta_2

                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-6] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-5] = L_shaft * beta_2 - beta_3
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)+1] = beta_3

                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-3] = beta_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-2] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)+3] = -beta_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)+4] = -beta_2

                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-3] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-2] = L_shaft * beta_2 - beta_3
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)+3] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)+4] = beta_3
                    # 轴向振动刚度
                    axial_k = E * area / L_shaft
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)-4] = axial_k
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)+2] = -axial_k
                    # 扭转振动刚度
                    Ip = np.pi * (shaft_segment_single[k, 2]**4 - shaft_segment_single[k, 3]**4) / 32
                    stiffness_twist = G * Ip / L_shaft
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)-1] = stiffness_twist
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)+5] = -stiffness_twist
 
                elif k == shaft_segment - 1:
                    # 弯曲振动刚度
                    I_section = np.pi * (shaft_segment_single[k-1, 2]**4 - shaft_segment_single[k-1, 3]**4) / 64
                    L_shaft = shaft_segment_single[k-1, 1]
                    E = shaft_segment_single[k-1, 4]
                    possion = shaft_segment_single[k-1, 5]
                    a_factor = shaft_segment_single[k-1, 6]
                    G = E / (2 * (1 + possion))
                    area = (np.pi / 4) * (shaft_segment_single[k-1, 2]**2 - shaft_segment_single[k-1, 3]**2)
                    if a_factor != 0:
                        f_sheer = 6 * E * I_section / (a_factor * G * area * L_shaft**2)
                    else:
                        f_sheer = 0
                    
                    beta_1 = 12 * E * I_section / (L_shaft**3 * (1 + 2 * f_sheer))
                    beta_2 = 0.5 * L_shaft * beta_1
                    beta_3 = L_shaft**2 * (1 - f_sheer) * beta_1 / 6
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-12] = -beta_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-11] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-6] = beta_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-5] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-12] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-11] = beta_3
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-6] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-5] = L_shaft * beta_2 - beta_3
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-9] = -beta_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-8] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-3] = beta_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-2] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-9] = -beta_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-8] = beta_3
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-3] = beta_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-2] = L_shaft * beta_2 - beta_3

                    # 轴向振动刚度
                    axial_k = E * area / L_shaft
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)-10] = -axial_k
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)-4] = axial_k

                    # 扭转振动刚度
                    Ip = np.pi * (shaft_segment_single[k-1, 2]**4 - shaft_segment_single[k-1, 3]**4) / 32
                    stiffness_twist = G * Ip / L_shaft
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)-7] = -stiffness_twist
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)-1] = stiffness_twist

                else:
                    # 弯曲振动刚度
                    I_section_1 = np.pi * (shaft_segment_single[k-1, 2]**4 - shaft_segment_single[k-1, 3]**4) / 64
                    L_shaft_1 = shaft_segment_single[k-1, 1]
                    E_1 = shaft_segment_single[k-1, 4]
                    possion_1 = shaft_segment_single[k-1, 5]
                    a_factor_1 = shaft_segment_single[k-1, 6]
                    G_1 = E_1 / (2 * (1 + possion_1))
                    area_1 = (np.pi / 4) * (shaft_segment_single[k-1, 2]**2 - shaft_segment_single[k-1, 3]**2)
                    if a_factor_1 != 0:
                        f_sheer_1 = 6 * E_1 * I_section_1 / (a_factor_1 * G_1 * area_1 * L_shaft_1**2)
                    else:
                        f_sheer_1=0
                    
                    beta_1_1 = 12 * E_1 * I_section_1 / (L_shaft_1**3 * (1 + 2 * f_sheer_1))
                    beta_2_1 = 0.5 * L_shaft_1 * beta_1_1
                    beta_3_1 = L_shaft_1**2 * (1 - f_sheer_1) * beta_1_1 / 6
                    I_section_2 = np.pi * (shaft_segment_single[k, 2]**4 - shaft_segment_single[k, 3]**4) / 64
                    L_shaft_2 = shaft_segment_single[k, 1]
                    E_2 = shaft_segment_single[k, 4]
                    possion_2 = shaft_segment_single[k, 5]
                    a_factor_2 = shaft_segment_single[k, 6]
                    G_2 = E_2 / (2 * (1 + possion_2))
                    area_2 = (np.pi / 4) * (shaft_segment_single[k, 2]**2 - shaft_segment_single[k, 3]**2)
                    if a_factor_2 != 0:
                        f_sheer_2 = 6 * E_2 * I_section_2 / (a_factor_2 * G_2 * area_2 * L_shaft_2**2)
                    else:
                        f_sheer_2 = 0

                    # 轴向振动刚度
                    beta_1_2 = 12 * E_2 * I_section_2 / (L_shaft_2**3 * (1 + 2 * f_sheer_2))
                    beta_2_2 = 0.5 * L_shaft_2 * beta_1_2
                    beta_3_2 = L_shaft_2**2 * (1 - f_sheer_2) * beta_1_2 / 6
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-12] = -beta_1_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-11] = -beta_2_1
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-6] = beta_1_1 + beta_1_2
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)-5] = -beta_2_1 + beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)] = -beta_1_2
                    shaft_single_stiffness_local[6*(k+1)-6, 6*(k+1)+1] = beta_2_2

                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-12] = beta_2_1
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-11] = beta_3_1
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-6] = -beta_2_1 + beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)-5] = (L_shaft_1 * beta_2_1 - beta_3_1) + (L_shaft_2 * beta_2_2 - beta_3_2)
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)] = -beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-5, 6*(k+1)+1] = beta_3_2

                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-9] = -beta_1_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-8] = beta_2_1
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-3] = beta_1_1 + beta_1_2
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)-2] = beta_2_1 - beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)+3] = -beta_1_2
                    shaft_single_stiffness_local[6*(k+1)-3, 6*(k+1)+4] = -beta_2_2

                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-9] = -beta_2_1
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-8] = beta_3_1
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-3] = beta_2_1 - beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)-2] = (L_shaft_1 * beta_2_1 - beta_3_1) + (L_shaft_2 * beta_2_2 - beta_3_2)
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)+3] = beta_2_2
                    shaft_single_stiffness_local[6*(k+1)-2, 6*(k+1)+4] = beta_3_2

                    # 轴向振动刚度
                    axial_k_1 = E_1 * area_1 / L_shaft_1
                    axial_k_2 = E_2 * area_2 / L_shaft_2
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)-10] = -axial_k_1
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)-4] = axial_k_1 + axial_k_2
                    shaft_single_stiffness_local[6*(k+1)-4, 6*(k+1)+2] = -axial_k_2

                    # 扭转振动刚度
                    Ip_1 = np.pi * (shaft_segment_single[k-1, 2]**4 - shaft_segment_single[k-1, 3]**4) / 32
                    stiffness_twist_1 = G_1 * Ip_1 / L_shaft_1
                    Ip_2 = np.pi * (shaft_segment_single[k, 2]**4 - shaft_segment_single[k, 3]**4) / 32
                    stiffness_twist_2 = G_2 * Ip_2 / L_shaft_2
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)-7] = -stiffness_twist_1
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)-1] = stiffness_twist_1 + stiffness_twist_2
                    shaft_single_stiffness_local[6*(k+1)-1, 6*(k+1)+5] = -stiffness_twist_2

            shaft_single_stiffness_temp = np.zeros((6 * shaft_segment, 6 * shaft_segment))
            for ii in range(shaft_segment):
                for jj in range(shaft_segment):
                    shaft_single_stiffness_temp[6*ii, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii, 6*jj:6*(jj+1)]
                    shaft_single_stiffness_temp[6*ii+1, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii+3, 6*jj:6*(jj+1)]
                    shaft_single_stiffness_temp[6*ii+2, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii+2, 6*jj:6*(jj+1)]
                    shaft_single_stiffness_temp[6*ii+3, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii+4, 6*jj:6*(jj+1)]
                    shaft_single_stiffness_temp[6*ii+4, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii+1, 6*jj:6*(jj+1)]
                    shaft_single_stiffness_temp[6*ii+5, 6*jj:6*(jj+1)] = shaft_single_stiffness_local[6*ii+5, 6*jj:6*(jj+1)]

                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj]
                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj+1] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj+3]
                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj+2] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj+2]
                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj+3] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj+4]
                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj+4] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj+1]
                    shaft_single_stiffness[6*ii:6*(ii+1), 6*jj+5] = shaft_single_stiffness_temp[6*ii:6*(ii+1), 6*jj+5]


            addition_stiffness = np.zeros((6, 6))
            addition_stiffness[0, 0] = 10**3
            addition_stiffness[1, 1] = 10**3
            addition_stiffness[2, 2] = 10**3
            addition_stiffness[3, 3] = 10**5
            addition_stiffness[4, 4] = 10**5
            addition_stiffness[5, 5] = 10**5
            shaft_single_stiffness[:6, :6] = addition_stiffness

            self.roller_screw_stiffness[dd:dd+6*shaft_segment, 0] = order_shaft
            self.roller_screw_stiffness[dd:dd+6*shaft_segment,1:1+6*shaft_segment] = shaft_single_stiffness
            dd = dd + 6 * shaft_segment
 


    def compute_influence_coefficients_outer(self, H, indices):
        """
        计算影响系数 C_a, C_b, C_delta
        :param H: 插值数据表 (2D NumPy 数组)，第一列是 x 值，后续列是 y 值
        :param p_values: 需要插值的 p 值 (NumPy 数组)
        :param indices: H 中对应的 y 值索引 (从 1 开始计数)
        :return: 计算得到的 C_a, C_b, C_delta (NumPy 数组)
        """
        results = []
        for idx in indices:
            interp_func = interp1d(H[:, 0], H[:, idx - 1], kind='cubic', fill_value="extrapolate")
            results.append(interp_func(self.p_outer_f))
        return results
    
    def compute_influence_coefficients_inner(self, H, indices):
        """
        计算影响系数 C_a, C_b, C_delta
        :param H: 插值数据表 (2D NumPy 数组)，第一列是 x 值，后续列是 y 值
        :param p_values: 需要插值的 p 值 (NumPy 数组)
        :param indices: H 中对应的 y 值索引 (从 1 开始计数)
        :return: 计算得到的 C_a, C_b, C_delta (NumPy 数组)
        """
        results = []
        for idx in indices:
            interp_func = interp1d(H[:, 0], H[:, idx - 1], kind='cubic', fill_value="extrapolate")
            results.append(interp_func(self.p_inner_f))
        return results

    def circumcenter(self, tri):
        """计算三角形的外接圆圆心
        """
        v0 = tri[2] - tri[1]
        v1 = tri[0] - tri[2]
        v2 = tri[1] - tri[0]
        area = (v1[0]*v2[1]-v1[1]*v2[0])/ 2.0

        x = np.sum(tri**2, axis=1)
        w0 = x[2] + x[1]
        w1 = x[0] + x[2]
        w2 = x[1] + x[0]

        W = np.array([[0, -1], [1, 0]], dtype=np.float64)
        fe0 = w0 * v0 @ W
        fe1 = w1 * v1 @ W
        fe2 = w2 * v2 @ W
        c = 0.25 * (fe0 + fe1 + fe2) / area
        r = np.sqrt(np.sum((c - tri[0]) ** 2))
        return c, r 


    def add_plot(self, axes):
        axes.axis('equal')
        axes.grid(True, linestyle='--', color='gray')
        p0 = self._mesh_outer_data[:, 3:5]
        p1 = self._mesh_inner_data[:, 3:5]
        p = np.vstack((p0, p1))
        axes.plot(p[:, 0], p[:, 1], 'ro')

        # 画丝杠外啮合圆
        r = self._data[0, 43-1] / 2
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        axes.plot(x, y, 'k-')

        # 画螺母内啮合圆
        r = self._data[0, 44-1] / 2 
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        axes.plot(x, y, 'b-')
        
        # 绘制滚柱圆
        r = self._data[0, 45-1] / 2
        theta = np.linspace(0, 2 * np.pi, self.number_roller, endpoint=False)
        roller_centers_x = self.center_roller * np.cos(theta)
        roller_centers_y = self.center_roller * np.sin(theta)
        theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
        for cx, cy in zip(roller_centers_x, roller_centers_y):
            x = r * np.cos(theta) + cx
            y = r * np.sin(theta) + cy
            axes.plot(x, y, 'b-')  


# 获取当前脚本 `prs.py` 所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))  
mat_file_path = os.path.join(script_dir, 'potential_roller_screw.mat')

# 读取 .mat 文件
prs_data = sio.loadmat(mat_file_path)

type_load = int(prs_data['type_load'][0].item())
order_load = int(prs_data['order_load'][0].item())
data = prs_data['roller_screw_initial']
modify_nut = prs_data['modify_nut']
modify_screw = prs_data['modify_screw']
norm_roller_screw = prs_data['norm_roller_screw']
mesh_inner_data = prs_data['mesh_inner_matrix']
mesh_outer_data = prs_data['mesh_outer_matrix']

prs = PlanetaryRollerScrew(order_load, data, mesh_outer_data, mesh_inner_data, norm_roller_screw, modify_nut, modify_screw, type_load)

# 计算外啮合影响系数
prs.C_a_outer, prs.C_b_outer, prs.C_delta_outer = prs.compute_influence_coefficients_outer(H, [2, 3, 4])

# 计算内啮合影响系数
prs.C_a_inner, prs.C_b_inner, prs.C_delta_inner = prs.compute_influence_coefficients_inner(H, [2, 3, 4])

# 计算行星滚柱丝杠内螺纹 / 外螺纹的啮合柔度
prs.flexible_roller_screw()

# 求解行星滚柱丝杠各接触点的受载法向载荷
prs.solver()

# 计算啮合接触半轴长
prs.compute_half_width()
# 计算接触应力
prs.compute_stress()
# 计算接触变形量
prs.computer_delta()
# 计算载荷分布均方值
prs.compute_s_load()
# 计算行星滚柱丝杠总势能
prs.potential_roller_screw()
# 保存结果
prs.result={
    'potential_total': prs.potential_total, # 行星滚柱丝杠总的势能
    'Fn_nut': prs.nut.Fn, # 螺母的接触点法向载荷的向量 [1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
    'Fn_screw': prs.screw.Fn, # 丝杠的接触点法向载荷的向量 [1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
    'flexible_inner': prs.flexible_inner, # 内螺纹的啮合柔度
    'flexible_outer': prs.flexible_outer, # 外螺纹的啮合柔度
    'B_inner': prs.B_inner, # 内啮合压缩势能辅助值
    'B_outer': prs.B_outer, # 外啮合压缩势能辅助值
    'half_width_a_inner': prs.half_width_a_inner, # 内啮合的接触长半轴(mm)
    'half_width_b_inner': prs.half_width_b_inner, # 内啮合的接触短半轴(mm)
    'stress_inner': prs.stress_inner, # 内啮合接触应力(Mpa)
    'half_width_a_outer': prs.half_width_a_outer, # 外啮合的接触长半轴(mm)
    'half_width_b_outer': prs.half_width_b_outer, # 外啮合的接触短半轴(mm)
    'stress_outer': prs.stress_outer, # 外啮合接触应力(Mpa)
    'ratio_nut': prs.nut.ratio, # 螺母轴向分量比例
    'ratio_screw': prs.screw.ratio, # 丝杠轴向分量比例
    'distribution_nut': prs.distribution_nut, # 螺母的接触点法向载荷不均载系数的向量 [1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
    'distribution_screw': prs.distribution_screw, #丝杠的接触点法向载荷不均载系数的向量 [1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
    'delta_outer': prs.delta_outer, # 外啮合接触变形量
    'delta_inner': prs.delta_inner, # 内啮合接触变形量
    'S_load_nut': prs.nut.s_load, # 螺母载荷分布均方值
    'S_load_screw': prs.screw.s_load, # 丝杠载荷分布均方值
}



# tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
# print(prs.circumcenter(tri))


fig = plt.figure()
axes = fig.add_subplot(111)
prs.add_plot(axes)
plt.show()
