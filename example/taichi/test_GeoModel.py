from fealpy.ti.GeoModel import GeoModel
import taichi as ti

square = GeoModel()
square.point(-1,-1.0,0.0,0)
square.point(1,-1,0.0,1)
square.point(1,1,0.0,2)
square.point(-1,1,0.0,3)

square.line(0,1,0)
square.line(1,2,1)
square.line(2,3,2)
square.line(3,0,3)

window = ti.ui.Window('square',(800,600))
points = square.map_to_01()
while window.running:
    square.show(window)
    window.show()
