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
        # gkm.display(p1, p2, p3, line, arc, arc2, spline1, spline2)
        gkm.display(spline1)

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_entity_construct2(self, input_data, kernel):
        gkm.set_adapter(kernel)

        box = gkm.add_box(2, 2, 2, 1, 1, 1)
        ellipsoid = gkm.add_ellipsoid(0, 0, 0, 5, 3, 2)
        cylinder1 = gkm.add_cylinder(0, 0, 0, 1, 4)
        cylinder2 = gkm.add_cylinder(-2, -2, -2, 1, 4, axis=(1, 0, 0))
        ring = gkm.add_ring(10, 10, 0, 1, 2)
        torus = gkm.add_torus(10, 10, 0, 10, 2)
        hollow_cyl = gkm.add_hollow_cylinder(0, 0, 0, 5, 3, 10)


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
        gkm.set_adapter(kernel)

        ori = gkm.add_point(0, 0, 0)
        box_base = gkm.add_box(-12.5, -12.5, -0.1, 25, 25, 0.1)

        box1 = gkm.add_box(-0.12, -0.06, 0.0, 0.24, 0.12, 0.6)
        box2 = gkm.translate(box1, (0.4, 0, 0))
        box3 = gkm.translate(gkm.rotate(box1, (0, 0, 0), (0, 0, 1), 3.14/4), (0, 0.4, 0))

        total_shape = gkm.boolean_union(box_base, box1, box2, box3)

        gkm.display(ori, total_shape)

    @pytest.mark.parametrize("kernel", ['occ'])
    @pytest.mark.parametrize("input_data", geometry_data)
    def test_planetary_roller_screw(self, input_data, kernel):
        gkm.set_adapter(kernel)

        screw = gkm.add_cylinder(0, 0, 0, 18/2, 240)
        nut = gkm.add_hollow_cylinder(0, 0, 0, 45/2, 35/2, 100)
        roller = gkm.add_cylinder(13, 0, 0, 5.5/2, 100)

        total_shape = gkm.boolean_union(screw, nut, roller)
        gkm.display(total_shape)



if __name__ == "__main__":
    pytest.main(["./test_geometry_base.py", "-k", "test_load_kernel"])