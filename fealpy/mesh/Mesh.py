from typing import Optional, Generic, TypeVar
from numpy.typing import NDArray
import numpy as np
from matplotlib import colors
from matplotlib import cm

# Descriptor for entities
_VT = TypeVar('_VT')
class Redirector(Generic[_VT]):
    def __init__(self, target: str) -> None:
        self._target = target

    def __get__(self, obj, objtype) -> _VT:
        return getattr(obj, self._target)

    def __set__(self, obj, val: _VT):
        setattr(obj, self._target, val)

class MeshDataStructure():
    NN: int = -1
    TD: int

    cell: NDArray 
    face: Optional[NDArray]
    edge: Optional[NDArray]
    edge2cell: Optional[NDArray]

    localEdge: NDArray
    localFace: NDArray
    localCell: NDArray

    NVC: int
    NVE: int
    NVF: int
    NEC: int
    NFC: int

    def construct(self):
        raise NotImplementedError

    def number_of_cells(self):
        """Number of cells"""
        return self.cell.shape[0]

    def number_of_faces(self):
        """Number of faces"""
        return self.face.shape[0]

    def number_of_edges(self):
        """Number of edges"""
        return self.edge.shape[0]

    def number_of_nodes(self):
        """Number of nodes"""
        return self.NN

class Mesh:

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NF

    def number_of_cells(self):
        return self.ds.NC

    def geo_dimension(self):
        raise NotImplementedError

    def top_dimension(self):
        raise NotImplementedError

    def uniform_refine(self):
        raise NotImplementedError

    def integrator(self, k):
        raise NotImplementedError

    def number_of_entities(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_barycenter(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def entity_measure(self, etype, index=np.s_[:]):
        raise NotImplementedError

    def shape_function(self, p):
        raise NotImplementedError

    def grad_shape_function(self, p, index=np.s_[:]):
        raise NotImplementedError
    
    def number_of_local_ipoints(self, p):
        raise NotImplementedError

    def number_of_global_ipoints(self, p):
        raise NotImplementedError

    def interpolation_points(self, p, index=np.s_[:]):
        raise NotImplementedError

    def cell_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def edge_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def face_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def node_to_ipoint(self, p, index=np.s_[:]):
        raise NotImplementedError

    def show_angle(self, axes, angle=None):
        """
        @brief 显示网格角度的分布直方图
        """
        if angle is None:
            angle = self.angle() 
        hist, bins = np.histogram(angle.flatten('F')*180/np.pi, bins=50, range=(0, 180))
        center = (bins[:-1] + bins[1:])/2
        axes.bar(center, hist, align='center', width=180/50.0)
        axes.set_xlim(0, 180)
        mina = np.min(angle.flatten('F')*180/np.pi)
        maxa = np.max(angle.flatten('F')*180/np.pi)
        meana = np.mean(angle.flatten('F')*180/np.pi)
        axes.annotate('Min angle: {:.4}'.format(mina), xy=(0.41, 0.5),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max angle: {:.4}'.format(maxa), xy=(0.41, 0.45),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average angle: {:.4}'.format(meana), xy=(0.41, 0.40),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        return mina, maxa, meana

    def show_quality(self, axes, qtype=None, quality=None):
        """
        @brief 显示网格质量分布的分布直方图
        """
        if quality is None:
            quality = self.cell_quality() 
        minq = np.min(quality)
        maxq = np.max(quality)
        meanq = np.mean(quality)
        hist, bins = np.histogram(quality, bins=50, range=(0, 1))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=0.02)
        axes.set_xlim(0, 1)
        axes.annotate('Min quality: {:.6}'.format(minq), xy=(0.1, 0.5),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max quality: {:.6}'.format(maxq), xy=(0.1, 0.45),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average quality: {:.6}'.format(meanq), xy=(0.1, 0.40),
                textcoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')
        return minq, maxq, meanq

    def add_plot(self):
        raise NotImplementedError

    def error(self, u, v, q=3, power=2, celltype=False):
        """
        @brief 给定两个函数，计算两个函数的之间的差，默认计算 L2 差（power=2) 

        """
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(u):
            if not hasattr(u, 'coordtype'): 
                u = u(ps)
            else:
                if u.coordtype == 'cartesian':
                    u = u(ps)
                elif u.coordtype == 'barycentric':
                    u = u(bcs)

        if callable(v):
            if not hasattr(v, 'coordtype'):
                v = v(ps)
            else:
                if v.coordtype == 'cartesian':
                    v = v(ps)
                elif v.coordtype == 'barycentric':
                    v = v(bcs)

        if u.shape[-1] == 1:
            u = u[..., 0]

        if v.shape[-1] == 1:
            v = v[..., 0]

        cm = self.entity_measure('cell')

        f = np.power(np.abs(u - v), power) 
        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                if f.shape[1] != len(cm):
                    f = f.transpose(0,2,1)
                e = np.einsum('q, qc..., c->c...', ws, f, cm)

        if celltype is False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )


    def find_node(self, axes, node=None, 
            index=np.s_[:],
            showindex=False, 
            color='r', markersize=20, 
            fontsize=16, fontcolor='r', 
            multiindex=None,  add_y_axis=False):

        if node is None:
            node = self.entity('node')

        GD = self.geo_dimension()

        # 将一维节点映射到二维平面上
        if GD == 1:
            # node = np.r_['1', node, np.zeros_like(node)]
            if add_y_axis:
                node = np.r_['1', node, np.full_like(node, 0.5)]
            else:
                node = np.r_['1', node, np.zeros_like(node)]
            GD = 2

        if isinstance(index, slice) and  index == np.s_[:]:
            index = range(node.shape[0])
        elif (type(index) is np.int_):
            index = np.array([index], dtype=np.int_)
        elif (type(index) is np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
        elif (type(index) is list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)

        if (type(color) is np.ndarray) and (np.isreal(color[0])):
            umax = color.max()
            umin = color.min()
            norm = colors.Normalize(vmin=umin, vmax=umax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            color = mapper.to_rgba(color)

        bc = node[index]
        if GD == 2:
            axes.scatter(bc[..., 0], bc[..., 1], c=color, s=markersize)
            if showindex:

                if multiindex is not None:
                    if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                        for i, idx in enumerate(multiindex):
                            s = str(idx).replace('[', '(')
                            s = s.replace(']', ')')
                            s = s.replace(' ', ',')
                            axes.text(bc[i, 0], bc[i, 1], s,
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor)
                    else:
                        for i, idx in enumerate(multiindex):
                            axes.text(bc[i, 0], bc[i, 1], str(idx),
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor) 
                else:
                    for i in range(len(index)):
                        axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                                multialignment='center', fontsize=fontsize, 
                                color=fontcolor) 
        else:
            axes.scatter(bc[..., 0], bc[..., 1], bc[..., 2], c=color, s=markersize)
            if showindex:
                if multiindex is not None:
                    if (type(multiindex) is np.ndarray) and (len(multiindex.shape) > 1):
                        for i, idx in enumerate(multiindex):
                            s = str(idx).replace('[', '(')
                            s = s.replace(']', ')')
                            s = s.replace(' ', ',')
                            axes.text(bc[i, 0], bc[i, 1], bc[i, 2], s,
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor)
                    else:
                        for i, idx in enumerate(multiindex):
                            axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(idx),
                                    multialignment='center',
                                    fontsize=fontsize, 
                                    color=fontcolor) 
                else:
                    for i in range(len(index)):
                        axes.text(bc[i, 0], bc[i, 1], bc[i, 2], str(index[i]),
                                 multialignment='center', fontsize=fontsize, color=fontcolor) 

    def find_edge(self, axes, 
            index=np.s_[:], 
            showindex=False,
            color='g', markersize=22,
            fontsize=18, fontcolor='g'):
        return self.find_entity(axes, 'edge', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)

    def find_face(self, axes, 
            index=np.s_[:], 
            showindex=False,
            color='b', markersize=24,
            fontsize=20, fontcolor='b'):
        return self.find_entity(axes, 'face', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)

    def find_cell(self, axes, 
            index=np.s_[:], 
            showindex=False,
            color='y', markersize=26,
            fontsize=22, fontcolor='y'):
        return self.find_entity(axes, 'cell', 
                showindex=showindex,
                color=color, 
                markersize=markersize,
                fontsize=fontsize, 
                fontcolor=fontcolor)

    def find_entity(self, axes, 
            etype, 
            index=np.s_[:], 
            showindex=False,
            color='r', markersize=20,
            fontsize=24, fontcolor='k'):

        GD = self.geo_dimension()
        bc = self.entity_barycenter(etype, index=index)

        if GD == 1:
            bc = np.r_['1', bc, np.zeros_like(bc)]
            GD = 2
        if isinstance(index, slice) and index == np.s_[:]:
            index = range(bc.shape[0])
        elif (type(index) is np.int_):
            index = np.array([index], dtype=np.int_)
        elif (type(index) is np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
        elif (type(index) is list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)

        if (type(color) is np.ndarray) & (np.isreal(color[0])):
            umax = color.max()
            umin = color.min()
            norm = colors.Normalize(vmin=umin, vmax=umax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            color = mapper.to_rgba(color)

        bc = bc[index]
        if GD == 2:
            axes.scatter(bc[:, 0], bc[:, 1], c=color, s=markersize)
            if showindex:
                for i in range(len(index)):
                    axes.text(bc[i, 0], bc[i, 1], str(index[i]),
                            multialignment='center', fontsize=fontsize, 
                            color=fontcolor) 
        else:
            axes.scatter(bc[:, 0], bc[:, 1], bc[:, 2], c=color, s=markersize)
            if showindex:
                for i in range(len(index)):
                    axes.text(
                            bc[i, 0], bc[i, 1], bc[i, 2],
                            str(index[i]),
                            multialignment='center',
                            fontsize=fontsize, color=fontcolor)
