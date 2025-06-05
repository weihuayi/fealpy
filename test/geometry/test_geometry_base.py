import pytest

from fealpy.backend import backend_manager as bm
from fealpy.geometry import geometry_kernel_manager as gkm
from fealpy.utils import timer

from geometry_base_data import *

class TestGeometryKernelBase:

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_load_kernel(self, input_data, kernel):
        gkm.set_adapter(kernel)

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_display(self, input_data, kernel):
        gkm.set_adapter(kernel)

        box1 = gkm.add_rectangle(0, 0, 0, 5, 10)
        box2 = gkm.add_rectangle(5, 5, 0, 5, 10)

        gkm.display(box1, box2, color=["blue", "red"], transpose=[0.3, 0.6])


    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_entity_construct(self, input_data, kernel):
        gkm.set_adapter(kernel)

        # 点
        p1 = gkm.add_point(0, 0, 0)
        p2 = gkm.add_point(5, 5, 0)
        p3 = gkm.add_point(10, 0, 0)
        p4 = gkm.add_point(0, 10, 0)
        # 线
        line = gkm.add_line(p1, p2)
        # 圆弧
        arc = gkm.add_arc(p1, p2, p3)
        arc2 = gkm.add_arc_center([0, 0, 0], [2, 0, 0], [-1, 0, 0])
        # 样条曲线
        ctrl_points1 = [(0, 0, 0), (2, 3, 1), (5, 4, 2), (7, 1, 3)]
        spline1 = gkm.add_spline(ctrl_points1)
        spline2 = gkm.add_spline([p1, p2, p3, p4])
        # 显示
        gkm.display(p1, p2, p3, line, arc, arc2, spline1, spline2,
                    color=["blue", "red", "green", "yellow", "purple", "orange"],
                    transpose=[0.3, 0.6, 0.9, 0.5, 0.7, 0.8],)
        # gkm.display(spline1)

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_entity_construct2(self, input_data, kernel):
        gkm.set_adapter(kernel)

        box = gkm.add_box(2, 10, 2, 1, 1, 1)
        ellipsoid = gkm.add_ellipsoid(-10, 0, 0, 5, 3, 2)
        cylinder1 = gkm.add_cylinder(5, 5, 0, 1, 4)
        cylinder2 = gkm.add_cylinder(-5, -5, -2, 1, 4, axis=(1, 0, 0))
        torus = gkm.add_torus(10, 10, 0, 10, 2)
        hollow_cyl = gkm.add_hollow_cylinder(5, -6, 0, 5, 3, 10)
        gkm.display(box, ellipsoid, cylinder1, cylinder2, torus, hollow_cyl,
                    color=["blue", "red", "green", "k", "purple", "orange"])

        edges1 = [gkm.add_line((0, 0, 0), (5, 0, 0)),
                 gkm.add_line((5, 0, 0), (5, 5, 0)),
                 gkm.add_line((5, 5, 0), (0, 5, 0)),
                 gkm.add_line((0, 5, 0), (0, 0, 0))]
        edge_loop1 = gkm.add_curve_loop(edges1)
        face1 = gkm.add_surface(edge_loop1)
        edges2 = [gkm.add_line((0, 0, 5), (5, 0, 5)),
                  gkm.add_line((5, 0, 5), (5, 5, 5)),
                  gkm.add_line((5, 5, 5), (0, 5, 5)),
                  gkm.add_line((0, 5, 5), (0, 0, 5))]
        edge_loop2 = gkm.add_curve_loop(edges2)
        face2 = gkm.add_surface(edge_loop2)
        edges3 = [gkm.add_line((0, 0, 0), (0, 0, 5)),
                  gkm.add_line((0, 0, 5), (5, 0, 5)),
                  gkm.add_line((5, 0, 5), (5, 0, 0)),
                  gkm.add_line((5, 0, 0), (0, 0, 0))]
        edge_loop3 = gkm.add_curve_loop(edges3)
        face3 = gkm.add_surface(edge_loop3)
        edges4 = [gkm.add_line((0, 5, 0), (0, 5, 5)),
                  gkm.add_line((0, 5, 5), (5, 5, 5)),
                  gkm.add_line((5, 5, 5), (5, 5, 0)),
                  gkm.add_line((5, 5, 0), (0, 5, 0))]
        edge_loop4 = gkm.add_curve_loop(edges4)
        face4 = gkm.add_surface(edge_loop4)
        edges5 = [gkm.add_line((0, 0, 0), (0, 0, 5)),
                    gkm.add_line((0, 0, 5), (0, 5, 5)),
                    gkm.add_line((0, 5, 5), (0, 5, 0)),
                    gkm.add_line((0, 5, 0), (0, 0, 0))]
        edge_loop5 = gkm.add_curve_loop(edges5)
        face5 = gkm.add_surface(edge_loop5)
        edges6 = [gkm.add_line((5, 0, 0), (5, 0, 5)),
                  gkm.add_line((5, 0, 5), (5, 5, 5)),
                  gkm.add_line((5, 5, 5), (5, 5, 0)),
                  gkm.add_line((5, 5, 0), (5, 0, 0))]
        edge_loop6 = gkm.add_curve_loop(edges6)
        face6 = gkm.add_surface(edge_loop6)
        # gkm.display(face1, face2, face3, face4, face5, face6, color=["blue", "red", "green", "yellow", "purple", "orange"])
        face_loop = gkm.add_face_loop([face1, face2, face3, face4, face5, face6])
        box_solid = gkm.add_volume(face_loop)
        gkm.display(box_solid)

        rectangle = gkm.add_rectangle(0,0,0, 10, 5)
        face = gkm.add_surface(rectangle)

        gkm.display(face)


    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_entity_construct3(self, input_data, kernel):
        gkm.set_adapter(kernel)

        rectangle = gkm.add_rectangle(0, 0, 0, 3, 2)
        disk = gkm.add_disk(5, -2, 0, 2, 3)
        circle = gkm.add_circle(0, 5, 0, 3)
        ring = gkm.add_ring(0, -5, 0, 2, 3)

        gkm.display(rectangle, disk, circle, ring,
                    color=["blue", "red", "green", "yellow"])


    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_bool_operate(self, input_data, kernel):
        gkm.set_adapter(kernel)

        # box1 = gkm.add_box(0, 0, 0, 5, 5, 5)
        # box2 = gkm.add_box(-2, -2, -2, 4, 3, 3)
        #
        # union = gkm.boolean_union(box1, box2)
        # gkm.display(union)
        # cut = gkm.boolean_cut(box1, box2)
        # gkm.display(cut)
        # intersect = gkm.boolean_intersect(box1, box2)
        # gkm.display(intersect)
        # fragment = gkm.boolean_fragment(box1, box2)
        # gkm.display(fragment)
        box = gkm.add_box(0, 0, 0, 5, 5, 5)
        sphere = gkm.add_sphere(0, 0, 0, 3)

        union = gkm.boolean_union(box, sphere)
        gkm.display(union)
        cut = gkm.boolean_cut(box, sphere)
        gkm.display(cut)
        intersect = gkm.boolean_intersect(box, sphere)
        gkm.display(intersect)
        fragment = gkm.boolean_fragment(box, sphere)
        gkm.display(fragment)



    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_geometry_operate(self, input_data, kernel):
        gkm.set_adapter(kernel)

        box_ori = gkm.add_box(0, 0, 0, 3, 4, 5)
        box_trans = gkm.translate(box_ori, (3, 4, 5))
        box_rotation1 = gkm.rotate(box_ori, (0, 0, 0), (0, 0, 1), 3.14/2)
        box_rotation2 = gkm.rotate(box_trans, (0, 0, 0), (0, 0, 1), 3.14/2)
        box_rotation3 = gkm.rotate(box_ori, (0, 4, 0), (1, 0, 0), 3.14/4)


        gkm.display(box_ori, box_trans, box_rotation1, box_rotation2, box_rotation3,
                    color=["blue", "red", "green", "yellow", "purple"], transpose=0.5)


    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_shape_discrete(self, input_data, kernel):
        from fealpy.mesh import TriangleMesh
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        gkm.set_adapter(kernel)

        box = gkm.add_sphere(0, 0, 0, 5)

        mesh = gkm.shape_discrete(box, deflection=0.1)
        node = mesh[0]
        cell = mesh[1]

        tri_mesh = TriangleMesh(bm.array(node, dtype=bm.float64), bm.array(cell, dtype=bm.int32))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        tri_mesh.add_plot(ax)
        plt.show()



    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_example_metalenses(self, input_data, kernel):
        # 设置参数
        square_size = 45  # 正方形的边长
        circle_radius = 20  # 圆的半径
        box_length = 0.24  # 长方体的长
        box_width = 0.12  # 长方体的宽
        box_height = 0.6  # 长方体的高
        circle_radius = 20  # 圆的半径
        square_size = 0.4  # 内部正方形的边长
        spacing = 0.4  # 相邻正方形的中心间距
        # 存储正方形中心坐标
        square_centers = []
        # 遍历圆内区域
        num_squares_x = int(circle_radius / spacing)  # x方向正方形数量
        num_squares_y = int(circle_radius / spacing)  # y方向正方形数量
        for i in range(-num_squares_x, num_squares_x + 1):  # x方向
            for j in range(-num_squares_y, num_squares_y + 1):  # y方向
                # 计算正方形中心坐标
                center_x = i * spacing
                center_y = j * spacing
                # 检查正方形的四个角点是否都在圆内
                if (bm.sqrt((center_x - square_size / 2) ** 2 + (center_y - square_size / 2) ** 2) <= circle_radius) and \
                        (bm.sqrt(
                            (center_x + square_size / 2) ** 2 + (center_y - square_size / 2) ** 2) <= circle_radius) and \
                        (bm.sqrt(
                            (center_x - square_size / 2) ** 2 + (center_y + square_size / 2) ** 2) <= circle_radius) and \
                        (bm.sqrt(
                            (center_x + square_size / 2) ** 2 + (center_y + square_size / 2) ** 2) <= circle_radius):
                    # 存储满足条件的中心坐标
                    square_centers.append((center_x, center_y, 0))
        # 将列表转换为二维数组
        square_centers_array = bm.array(square_centers)
        # 定义旋转的角度
        # 参数定义
        f = 60  # 设计焦距
        Lambda = 0.98  # 入射光波长

        def rotate_angle(p):
            x = p[..., 0]
            y = p[..., 1]
            # 计算旋转角度
            theta = bm.pi / Lambda * (f - bm.sqrt(x ** 2 + y ** 2 + f ** 2))
            return theta

        def generate_square(center, length, width):
            # 计算最小坐标点
            min_coords = center - bm.array([length / 2, width / 2, 0])
            return min_coords

        # 旋转的角度
        angle = rotate_angle(square_centers_array)
        # 旋转正方形的最小点坐标
        min_coords = generate_square(square_centers_array, box_length, box_width)
        total_shape = []
        total_shape.append(gkm.add_box(-22.5, -22.5, -0.1, 45, 45, 0.1))
        option = bm.concatenate((min_coords, square_centers_array), axis=1)
        angle = angle.reshape(-1, 1)
        option1 = bm.concatenate((option, angle), axis=1)
        for i, j, k, m, n, l, angles in option1:
            box = gkm.rotate(gkm.add_box(i, j, k, 0.24, 0.12, 0.6),
                             (m, n, l), (0, 0, 1), angles)
            total_shape.append(box)

        gkm.display(*total_shape)
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder
        # 创建一个复合形状
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)

        # 将所有形状添加到复合形状中
        for shape in total_shape:
            builder.Add(compound, shape)

        # 初始化 STL 写入器
        stl_writer = StlAPI_Writer()
        stl_writer.SetASCIIMode(True)  # True 为 ASCII 格式，False 为二进制格式

        # 导出复合形状到 STL 文件
        stl_writer.Write(compound, "metalenses.stl")
        print("STL file 'metalenses.stl' has been created successfully.")

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_planetary_roller_screw(self, input_data, kernel):
        gkm.set_adapter(kernel)

        screw = gkm.add_cylinder(0, 0, 0, 18/2, 240)
        nut = gkm.add_hollow_cylinder(0, 0, 0, 45/2, 35/2, 100)
        roller = gkm.add_cylinder(13, 0, 0, 5.5/2, 100)

        total_shape = gkm.boolean_union(screw, nut, roller)
        gkm.display(total_shape)


    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_performance(self, input_data, kernel):
        gkm.set_adapter(kernel)

        tmr = timer()
        next(tmr)
        maxiter = 100
        boxs = []

        for i in range(maxiter):
            box = gkm.add_box(0, 0, 0, 1+i*1.5, 1, 1)
            boxs.append(box)
        tmr.send("生成 box 花费时间")

        total_shape = gkm.boolean_union(*boxs)
        tmr.send("生成联合体花费时间")

        gkm.display(total_shape)
        # gkm.display(*boxs)
        tmr.send("显示花费时间")
        next(tmr)

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_mult_input(self, input_data, kernel):
        # bm.set_backend("pytorch")
        gkm.set_adapter(kernel)

        # box
        # 单个输入
        box1 = gkm.add_box(0, 0, 0, 5, 5, 5)
        gkm.display(box1)

        # 列表
        data_list = [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 3, 3, 3]]
        boxes1 = gkm.add_box(data_list)
        gkm.display(boxes1, color="red")

        # 元组
        data_tuple = ((0, 0, 0, 1, 1, 1), (1, 1, 1, 1, 1, 1), (2, 2, 2, 3, 3, 3))
        boxes3 = gkm.add_box(data_tuple)
        gkm.display(boxes3, color="blue")

        # 数组
        data_array = bm.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 3, 3, 3]])
        boxes5 = gkm.add_box(data_array)
        gkm.display(*boxes5, color=["green", "r", "y"])

        # # add_rectangle
        # data_array = bm.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 3, 3]])
        # rectangles = gkm.add_rectangle(data_array)
        # gkm.display(rectangles, color="green")
        #
        # # add_disk
        # data_array = bm.array([[0, 0, 0, 1, 2], [1, 1, 1, 2, 3], [2, 2, 2, 3, 4]])
        # disks = gkm.add_disk(data_array)
        # gkm.display(disks, color="green")
        #
        # # add_circle
        # data_array = bm.array([[0, 0, 0, 1], [1, 1, 1, 2], [2, 2, 2, 3]])
        # circles = gkm.add_circle(data_array)
        # gkm.display(circles, color="green")
        #
        # # add_ring
        # data_array = bm.array([[0, 0, 0, 1, 2], [1, 1, 1, 2, 3], [2, 2, 2, 3, 4]])
        # rings = gkm.add_ring(data_array)
        # gkm.display(rings, color="green")
        #
        # # add_box
        # data_array = bm.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 3, 3, 3]])
        # boxes5 = gkm.add_box(data_array)
        # gkm.display(boxes5, color="green")
        #
        # # add_ellipsoid
        # data_array = bm.array([[0, 0, 0, 1, 2, 3], [4, 4, 4, 1.5, 2.5, 3.5], [8, 8, 8, 2, 3, 4]])
        # ellipsoids = gkm.add_ellipsoid(data_array)
        # gkm.display(ellipsoids, color="green")
        #
        # # add_sphere
        # data_array = bm.array([[0, 0, 0, 1], [3, 3, 3, 1.5], [6, 6, 6, 2]])
        # spheres = gkm.add_sphere(data_array)
        # gkm.display(spheres, color="green")
        #
        # # add_cylinder
        data_array = bm.array([[0, 0, 0, 1, 5],
                                            [1, 1, 1, 2, 10]])
        axis = bm.array([[1, 0, 0],
                                [0, 1, 0]])
        cylinders = gkm.add_cylinder(data_array, axis=axis)
        gkm.display(cylinders, color="green")
        #
        # # add_torus
        # data_array = bm.array([[0, 0, 0, 2, 1], [3, 3, 3, 3, 1.5], [8, 8, 8, 4, 2]])
        # toruses = gkm.add_torus(data_array)
        # gkm.display(toruses, color="green")
        #
        # # add_hollow_cylinder
        # data_array = bm.array([[0, 0, 0, 2, 1, 3], [4, 4, 4, 2.5, 1.5, 3.5], [8, 8, 8, 3, 2, 4]])
        # axis = bm.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # hollow_cylinders = gkm.add_hollow_cylinder(data_array, axis=axis)
        # gkm.display(hollow_cylinders, color="green")

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_import_export(self, input_data, kernel):
        gkm.set_adapter(kernel)

        box = gkm.add_box(2, 10, 2, 1, 1, 1)
        ellipsoid = gkm.add_ellipsoid(-10, 0, 0, 5, 3, 2)
        cylinder1 = gkm.add_cylinder(5, 5, 0, 1, 4)
        cylinder2 = gkm.add_cylinder(-5, -5, -2, 1, 4, axis=(1, 0, 0))
        torus = gkm.add_torus(10, 10, 0, 10, 2)
        hollow_cyl = gkm.add_hollow_cylinder(5, -6, 0, 5, 3, 10)
        total_shape = [box, ellipsoid, cylinder1, cylinder2, torus, hollow_cyl]

        gkm.export_step(*total_shape, filename="box.step")
        gkm.export_stl(*total_shape, filename="box.stl", resolution=0.1)
        gkm.export_brep(*total_shape, filename="box.brep")

        box1 = gkm.import_step("box.step")
        box2 = gkm.import_stl("box.stl")
        box3 = gkm.import_brep("box.brep")

        gkm.display(box1)
        gkm.display(box2)
        gkm.display(box3)

        metalenses = gkm.import_stl("metalenses.stl")
        gkm.display(metalenses)


if __name__ == "__main__":
    pytest.main(["./test_geometry_base.py", "-k", "test_load_kernel"])