from fealpy.geometry.signed_distance_function import dcircle
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.polygon_mesh import PolygonMesh

import argparse
import matplotlib.pyplot as plt

if __name__ =='__main__':
    ## 参数解析
    parser = argparse.ArgumentParser(description=
                                     """
                                     界面网格生成算例，正方形中包含一个圆形，在两者边界生成界面网格，
                                     有两种网格结果可供选择：
                                     多边形网格——'poly'、三角形网格——'tri'。
                                     """)
    parser.add_argument('--mesh_type',
                        default='poly', type=str,
                        help='生成网格类型，默认为多边形网格.')
    args = parser.parse_args()
    mesh_type = args.mesh_type
    # 定义表示圆的符号距离函数
    circle = lambda p: dcircle(p, [0.5, 0.5], 0.25)
    # 底层网格区域
    box = [0, 1, 0, 1]
    # 剖分段数
    nx = 10
    ny = 10

    if mesh_type=='poly':
        # 界面网格生成——多边形
        interface_mesh = PolygonMesh.interfacemesh_generator(box, nx, ny, circle)
    elif mesh_type=='tri':
        # 界面网格生成——三角形
        interface_mesh = TriangleMesh.interfacemesh_generator(box, nx, ny, circle)
    else:
        raise TypeError('请选择正确的网格类型')

    fig, axes = plt.subplots()
    interface_mesh.add_plot(axes)
    plt.show()