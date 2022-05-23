import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.data_oriented
class GeoModel:
    def __init__(self):
        self.Point = {}
        self.Line = {}
    
    def point(self,x,y,z,tag):
        self.Point[tag] = [x,y,z]
    
    def line(self,startTag,endTag,tag):
        self.Line[tag] = [startTag,endTag]
    
    def point_to_tifield(self,points):
        NN = len(points)
        tipoints = ti.Vector.field(3,dtype=ti.f32,shape=NN)
        tags = sorted(list(points.keys()))
        for i in range(NN):
            tipoints[i] = ti.Vector(self.Point[tags[i]])
        return tipoints
    
    def line_to_tifield(self,lines):
        NL = len(lines)
        tilines = ti.Vector.field(2,dtype = ti.i32,shape=NL)
        pointtag = np.array(sorted(list(self.Point.keys())))
        pointflag = np.arange(len(pointtag))
        for i in range(NL):
            start = pointflag[pointtag==self.Line[i][0]]
            tilines[i] = ti.Vector([pointflag[pointtag==self.Line[i][0]][0],pointflag[pointtag==self.Line[i][1]][0]])
        return tilines
    
    def show(self,window):
        Point = self.map_to_01()
        point = self.point_to_tifield(Point)
        line = self.line_to_tifield(self.Line)
        window.get_cursor_pos()
        canvas = window.get_canvas()
        canvas.circles(point,radius=0.001,color=(0.5,0.5,0.5))
        canvas.lines(point,0.01,line,color=(0.5,0.5,0.5))
    
    def map_to_01(self):
        npoints = np.zeros((len(self.Point),3))
        for i,j in zip(self.Point,range(len(self.Point))):
            npoints[j] = self.Point[i]
        if np.min(npoints[:,0]) <0:
            npoints[:,0] +=0.1-np.min(npoints[:,0])
        if np.min(npoints[:,1]) <0:
            npoints[:,1] +=0.1-np.min(npoints[:,1])
        if np.min(npoints[:,0]) >1:
            npoints[:,0] -= np.min(npoints[:,0])-0.1
        if np.min(npoints[:,1]) >1:
            npoints[:,1] -= np.min(npoints[:,1])-0.1
        if np.max(npoints)>1:
            npoints = npoints*(1/(np.max(npoints)+1))
        Point = self.Point
        for i,j in zip(Point,range(len(self.Point))):
            Point[i] = npoints[j] 
        return Point

