
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

import taichi as ti

ti.init(arch=ti.cuda)


box = [0, 1, 0, 1]
mesh = TriangleMesh.from_box(box, nx=10, ny=10)
NN = mesh.number_of_nodes()
NE = mesh.number_of_edges()
NC = mesh.number_of_cells()

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

window = ti.ui.Window("PlotTriangleMesh", (512, 512))
canvas = window.get_canvas()

vertices = ti.Vector.field(2, float, NN)
vertices.from_numpy(bm.to_numpy(node))

tris = ti.field(ti.i32, NC*3)
tris.from_numpy(bm.to_numpy(cell.reshape(-1)))

lines = ti.field(ti.i32, NE*2) 
lines.from_numpy(bm.to_numpy(edge.reshape(-1)))
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1,))


def paint_mouse_ball():
    mouse = window.get_cursor_pos()
    mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])


def render():
    canvas.set_background_color((0, 0, 0))
    paint_mouse_ball()
    canvas.triangles(vertices, color=(1, 0.6, 0.2), indices=tris)
    canvas.lines(vertices, 0.001, indices=lines, color=(0, 0, 0))
    canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.02)


def main():
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False

        mouse_pos = window.get_cursor_pos()
        render()
        window.show()

if __name__ == "__main__":
    main()









