import torch
import numpy as np  

def Line_GeneralEquation(line=[1, 1, 1, 2]):
    """直线一般式"""
    A = line[3] - line[1]
    B = line[0] - line[2]
    C = line[2] * line[1] - line[0] * line[3]
    line = np.array([A, B, C])
    if B != 0:
        line = line / B
    return line

def SamplePointsOnLineSegment(point1, point2, distence):
    """在一条线段上均匀采样"""
    line_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  # 线段长度
    num = round(line_dist / distence)  # 采样段数量
    line = [point1[0], point1[1], point2[0], point2[1]]  # 两点成线
    line_ABC = Line_GeneralEquation(line)  # 一般式规范化
    newP = []
    newP.append(point1)  # 压入首端点
    if num > 0:
        dxy = line_dist / num  # 实际采样距离
        for i in range(1, num):
            if line_ABC[1] != 0:
                alpha = np.arctan(-line_ABC[0])
                dx = dxy * np.cos(alpha)
                dy = dxy * np.sin(alpha)
                if point2[0] - point1[0] > 0:
                    newP.append([point1[0] + i * dx, point1[1] + i * dy])
                else:
                    newP.append([point1[0] - i * dx, point1[1] - i * dy])
            else:
                if point2[1] - point1[1] > 0:
                    newP.append([point1[0], point1[1] + i * dxy])
                else:
                    newP.append([point1[0], point1[1] - i * dxy])
    newP.append([point2[0], point2[1]])  # 压入末端点
    return np.array(newP)

def multiLine2Points(lineXY, distence):
    '''将所给点连线并首尾连接，构成多边形，对每条边进行均匀采样'''
    lineXY = np.array(lineXY)
    newPoints = []
    # 对所有线段进行处理
    for i in range(len(lineXY) - 1):
        newP = SamplePointsOnLineSegment(lineXY[i, :], lineXY[i + 1, :], distence)
        newPoints.extend(newP)
    # 连接首尾两点，再进行均匀采样
    newP = SamplePointsOnLineSegment(lineXY[-1, :], lineXY[0, :], distence)
    newPoints.extend(newP)
    newPoints = np.array(newPoints)
    # 删除重复端点
    delInd = []
    for i in range(len(newPoints) - 1):
        if (newPoints[i, :] == newPoints[i + 1, :]).all():
            delInd.append(i)
    newPoints = np.delete(newPoints, delInd, axis=0)
    return newPoints

def uniformed_nodes(left:float, right:float, num):
    
    points = np.array(
                 [[left, left],
                  [left, right],
                  [right, right],
                  [right, left]])

    newPoints = multiLine2Points(points, 4*(right-left)/num)
    unique_rows, _ = torch.unique(torch.tensor(newPoints), dim=0, return_counts=True)
    unique_nodes = unique_rows.clone().detach()
    return unique_nodes

# print(uniformed_nodes(-1,1,0.1).shape)


