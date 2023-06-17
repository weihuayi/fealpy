import numpy as np
from fealpy.mesh.level_set_function import Sphere
import plotly.offline as py
import plotly.figure_factory as FF
import fealpy.tools.colors as cs

surface = Sphere()
mesh =  surface.init_mesh()
mesh.uniform_refine(n=4, surface=surface)

cell = mesh.ds.cell
point = mesh.point

pi = np.pi
theta = np.arctan2(point[:,1], point[:,0])
theta = (theta >= 0)*theta + (theta <0)*(theta + 2*pi)             
c = np.sin(5*theta)
c = np.sum(c[cell], axis=1)/3
c = cs.val_to_color(c)

fig = FF.create_trisurf(
        x = point[:, 0], 
        y = point[:, 1],
        z = point[:, 2],
        show_colorbar = True,
        plot_edges=False,
        simplices=cell)

fig['data'][0]['facecolor'] = c 
py.plot(fig, filename='test')
