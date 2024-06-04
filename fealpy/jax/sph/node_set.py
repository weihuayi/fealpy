
import jax.numpy as jnp
from scipy.spatial import cKDTree 
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.mesh.uniform_mesh_3d import UniformMesh3d
import matplotlib.pyplot as plt

class NodeSet():

    def __init__(self, node):
        self.NN = node.shape[0]
        self.node = node
        self.nodedata = {}
        self.tree = cKDTree(node)
        self.is_boundary = jnp.zeros(self.number_of_node(), dtype=bool)

    def number_of_node(self):
        return self.NN 
    
    def dimension(self):
        return self.node.shape[1]
       
    def is_boundary_node(self):
        return self.is_boundary

    def add_node_data(self, name, dtype=jnp.float64):
        NN = self.NN
        self.nodedata.update({names: jnp.zeros(NN, dtypes) 
                              for names, dtypes in zip(name, dtype)})

    def set_node_data(self, name, val):
        self.nodedata[name][:] = val

    def add_plot(self, axes, color='k', markersize=20):
        axes.set_aspect('equal')
        return axes.scatter(self.node[..., 0], self.node[..., 1], c=color, s=markersize)
    
    def neighbors(self, h, points=None):
        if points is None:
            return self.tree.query_ball_point(self.node, h)
        else:
            return self.tree.query_ball_point(points, h)

    @classmethod
    def from_tgv_domain(cls, dx=0.02, dy=0.02):
        result = jnp.mgrid[0:1+dx:dx, 0:1+dy:dy].reshape(2, -1).T
        return cls(result)

    @classmethod
    def from_ringshaped_channel_domain(cls, dx=0.02, dy=0.02):
        #下
        dp0 = jnp.mgrid[0:1.38+dx:dx, -dy:dy:dy].reshape(2, -1).T
        dp3 = jnp.mgrid[2.82:4.2+dx:dx, -dy:dy:dy].reshape(2, -1).T
        
        start_diff0 = jnp.array([1,0])-jnp.array([2.1,1.1])
        start_angle0 = jnp.arctan2(start_diff0[1], start_diff0[0])
        end_diff0 = jnp.array([3.2,0])-jnp.array([2.1,1.1])
        end_angle0 = jnp.arctan2(end_diff0[1], end_diff0[0])
        if end_angle0 < start_angle0:
            end_angle0 += 2 * jnp.pi
        angles0 = jnp.arange(start_angle0, end_angle0+dx, dx)
        x0 = jnp.array([2.1,1.1])[0] + jnp.cos(angles0)
        y0 = jnp.array([2.1,1.1])[1] + jnp.sin(angles0)
        dp1 = jnp.stack([x0, y0], axis=-1)-jnp.array([0,0.4])
        dp2 = jnp.stack([x0, y0], axis=-1)-jnp.array([0,0.4+dx])
        
        dp = jnp.vstack((dp0,dp1,dp2,dp3))
        #上
        up0 = jnp.mgrid[0:1.38+dx:dx, 1:1+dy:dy].reshape(2, -1).T
        up3 = jnp.mgrid[2.82:4.2+dx:dx, 1:1+dy:dy].reshape(2, -1).T
        
        start_diff1 = jnp.array([1,0])-jnp.array([2.1,1.1])
        start_angle1 = jnp.arctan2(start_diff1[1], start_diff1[0])
        end_diff1 = jnp.array([3.2,0])-jnp.array([2.1,1.1])
        end_angle1 = jnp.arctan2(end_diff1[1], end_diff1[0])
        if end_angle1 < start_angle1:
            end_angle1 += 2 * jnp.pi
        angles1 = jnp.arange(start_angle1+jnp.pi, end_angle1+jnp.pi+dx, dx)
        x1 = jnp.array([2.1,1.1])[0] + jnp.cos(angles1)
        y1 = jnp.array([2.1,1.1])[1] + jnp.sin(angles1)
        up1 = jnp.stack([x1, y1], axis=-1)-jnp.array([0,0.8])
        up2 = jnp.stack([x1, y1], axis=-1)-jnp.array([0,0.8-dx])
        
        up = jnp.vstack((up0,up1,up2,up3))
        #芯型
        circumference = 2 * jnp.pi * 0.25
        num = int(circumference / dx)
        angles = jnp.linspace(0, 2 * jnp.pi, num, endpoint=False)
        x = jnp.array([2.1,0.55])[0] + 0.25 * jnp.cos(angles)
        y = jnp.array([2.1,0.55])[1] + 0.25 * jnp.sin(angles)
        round = jnp.stack([x, y], axis=-1)
        return cls(jnp.vstack((dp, up,round)))

    @classmethod
    def from_dam_break_domain(cls, dx=0.02, dy=0.02):
        pp = jnp.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T        
        # 下
        bp0 = jnp.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
        bp1 = jnp.mgrid[-dx/2:4+dx/2:dx, -dy/2:dy/2:dy].reshape(2, -1).T
        bp = jnp.vstack((bp0, bp1))
        # 左
        lp0 = jnp.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
        lp1 = jnp.mgrid[-dx/2:dx/2:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        lp = jnp.vstack((lp0, lp1))
        # 右
        rp0 = jnp.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
        rp1 = jnp.mgrid[4+dx/2:4+dx:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        rp = jnp.vstack((rp0, rp1))
        
        boundaryp = jnp.vstack((bp, lp, rp))
        node = jnp.vstack((pp, boundaryp))
        result = cls(node)
        result.is_boundary = result.is_boundary.at[pp.shape[0]:].set(True)
        return result

#node_set = NodeSet.from_tgv_domain()
#node_set = NodeSet.from_ringshaped_channel_domain()
#node_set = from_dam_break_domain()
'''
fig, ax = plt.subplots()
node_set.add_plot(ax)
plt.show()
'''