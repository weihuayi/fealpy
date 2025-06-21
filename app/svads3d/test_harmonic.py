import numpy as np
import matplotlib.pyplot as plt
import pickle

from fealpy.mesh import TriangleMesh
from app.svads3d.harmonic_map import *
#from tools import *

def normal_condition(mesh):
    nodes = mesh.entity('node')
    edges = mesh.entity('edge')

    bdEdge = mesh.ds.boundary_edge()
    bdNode = nodes[mesh.ds.boundary_node_index()]

    normal_vectors = np.zeros((bdNode.shape[0],) + (2,))

    for index, node in enumerate(mesh.ds.boundary_node_index()):
        connected_edges = []
        for edge in bdEdge:
            if node in edge:
                connected_edges.append(edge)
        if len(connected_edges) == 2:
            nodes_of_edge0 = nodes[connected_edges[0]]
            edge_vector0 = nodes_of_edge0[1] - nodes_of_edge0[0]
            nodes_of_edge1 = nodes[connected_edges[1]]
            edge_vector1 = nodes_of_edge1[1] - nodes_of_edge1[0]
            if np.abs(np.dot(edge_vector0, edge_vector1)) < 1e-2:
                connected_edges[1] = connected_edges[0]
        for edge in connected_edges:
            # TODO: 优化
            nodes_of_edge = nodes[edge]
            edge_vector = nodes_of_edge[1] - nodes_of_edge[0]
            normal_vector = np.array([edge_vector[1], -edge_vector[0]])
            normal_vector /= np.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)
            normal_vectors[index] += normal_vector
        normal_vectors[index] /= len(connected_edges)

    normal_vectors /= np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]

    return normal_vectors

def vector_to_cross(vh: np.ndarray) -> np.ndarray:
    theta = np.mod(np.arctan2(vh[..., 1], vh[..., 0]), 2 * np.pi) / 4
    if len(vh.shape) == 1:
        # shape = (vh.shape[0], 4, 2)
        shape = (1, 4, 2)
    else:
        shape = vh.shape[:-1] + (4, 2)
    ch = np.zeros(shape)
    ch[..., 0, 0] = np.cos(theta)
    ch[..., 0, 1] = np.sin(theta)
    ch[..., 1, 0] = np.cos(theta + np.pi / 2)
    ch[..., 1, 1] = np.sin(theta + np.pi / 2)
    ch[..., 2, 0] = np.cos(theta + np.pi)
    ch[..., 2, 1] = np.sin(theta + np.pi)
    ch[..., 3, 0] = np.cos(theta + np.pi * 3 / 2)
    ch[..., 3, 1] = np.sin(theta + np.pi * 3 / 2)

    return ch


def cross_to_vector(ch: np.ndarray) -> np.ndarray:
    if len(ch.shape) == 1:
        shape = (ch.shape[0], 2)
    else:
        shape = ch.shape[:-2] + (2,)
    vh = np.zeros(shape)
    t0 = np.arctan2(ch[..., 1], ch[..., 0])
    t1 = np.mod(t0, 2 * np.pi)
    t2 = np.min(t1, axis=-1)
    theta = 4 * np.min(np.mod(np.arctan2(ch[..., 1], ch[..., 0]), 2 * np.pi), axis=-1)
    vh[..., 0] = np.cos(theta)
    vh[..., 1] = np.sin(theta)

    return vh


def one_direction_to_cross(direction: np.ndarray) -> np.ndarray:
    theta = np.arctan2(direction[..., 1], direction[..., 0])
    if len(direction.shape) == 1:
        shape = (direction.shape[0], 4, 2)
    else:
        shape = direction.shape[:-1] + (4, 2)
    ch = np.zeros(shape)
    ch[..., 0, :] = direction
    ch[..., 1, 0] = np.cos(np.mod(theta + np.pi / 2, 2 * np.pi))
    ch[..., 1, 1] = np.sin(np.mod(theta + np.pi / 2, 2 * np.pi))
    ch[..., 2, 0] = np.cos(np.mod(theta + np.pi, 2 * np.pi))
    ch[..., 2, 1] = np.sin(np.mod(theta + np.pi, 2 * np.pi))
    ch[..., 3, 0] = np.cos(np.mod(theta + np.pi * 3 / 2, 2 * np.pi))
    ch[..., 3, 1] = np.sin(np.mod(theta + np.pi * 3 / 2, 2 * np.pi))

    return ch


def vector_field_plotter(axes, node, uh, color='k', scale=20, angles='xy', scale_units='xy'):
    '''
    标架场绘制器
    :param axes: 用于绘图的坐标轴
    :param node: 网格节点
    :param uh: 网格向量场
    :param color: 箭头颜色
    :param scale: 向量缩放大小
    :param angles: 箭头角度的参考无
    :param scale_units: 缩放大小的参考物
    :return:
    '''
    x = node[:, 0]
    y = node[:, 1]
    u1 = np.array(uh[:, 0])
    u2 = np.array(uh[:, 1])
    axes.quiver(x, y, u1, u2, color=color, scale=scale, angles=angles, scale_units=scale_units)


def cross_field_plotter(axes, node, ch, cross_type=0, cross_size=0.01):
    '''
    标架场绘制器
    :param axes: 用于绘图的坐标轴
    :param node: 网格节点
    :param ch: 网格标架场
    :param cross_type: 标架场类型，0——一种颜色，1——四种颜色
    :param cross_size: 标架场尺寸
    :return:
    '''
    if len(ch.shape) == 2:
        ch = ch[np.newaxis, ...]
        node = node[np.newaxis, ...]
    for i in range(ch.shape[0]):
        x, y = ch[i, 0, :]  # 取每个标架的一个向量作为位置
        angle = np.arctan2(y, x)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        cross_coords = np.array([[-cross_size, 0], [cross_size, 0], [0, -cross_size], [0, cross_size]])
        rotated_cross = np.dot(cross_coords, rotation_matrix.T)
        if cross_type == 0:
            axes.plot(rotated_cross[:2, 0] + node[i, 0], rotated_cross[:2, 1] + node[i, 1], c='b', linewidth=0.5)
            axes.plot(rotated_cross[2:, 0] + node[i, 0], rotated_cross[2:, 1] + node[i, 1], c='b', linewidth=0.5)
        else:
            axes.plot([rotated_cross[0, 0], 0] + node[i, 0], [rotated_cross[0, 1], 0] + node[i, 1], c='b',
                      linewidth=0.5)
            axes.plot([rotated_cross[1, 0], 0] + node[i, 0], [rotated_cross[1, 1], 0] + node[i, 1], c='g',
                      linewidth=0.5)
            axes.plot([rotated_cross[2, 0], 0] + node[i, 0], [rotated_cross[2, 1], 0] + node[i, 1], c='y',
                      linewidth=0.5)
            axes.plot([rotated_cross[3, 0], 0] + node[i, 0], [rotated_cross[3, 1], 0] + node[i, 1], c='m',
                      linewidth=0.5)



## 生成半圆区域边界的节点
theta = np.linspace(0, np.pi, 10)
x = np.cos(theta)
y = np.sin(theta)
node = np.array([x, y], dtype=np.float64).T
## 生成半圆区域的网格
mesh = TriangleMesh.from_polygon_gmsh(node, h=0.05)

## 边界点投影
idx = mesh.ds.boundary_node_index()
idx1 = idx[mesh.node[idx, 1]>1e-10]
mesh.node[idx1] = mesh.node[idx1]/np.linalg.norm(mesh.node[idx1], axis=1).reshape(-1, 1)

## 边界条件
boundary_condition = cross_to_vector(one_direction_to_cross(normal_condition(mesh)))

## 计算调和映射
data = HarmonicMapData(mesh, mesh.ds.boundary_node_index(), boundary_condition)
uh = sphere_harmonic_map(data)

## 画图
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='c', showaxis=True)
plt.quiver(mesh.node[:, 0], mesh.node[:, 1], uh[:, 0], uh[:, 1], color='black') 
plt.show()


