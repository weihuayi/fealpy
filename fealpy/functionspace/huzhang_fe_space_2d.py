from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh

from .lagrange_fe_space import LagrangeFESpace
from .space import FunctionSpace
from fealpy.decorator import barycentric, cartesian

_MT = TypeVar("_MT", bound=Mesh)

class HuZhangFESpace2D(FunctionSpace, Generic[_MT]):
    """
    Hu-Zhang finite element space in 2D.
    """
    def __init__(self, mesh: _MT, p: int=1):
        self.mesh = mesh
        self.p = p
        self.space = LagrangeFESpace(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()
        self.dof = self.space.dof

        self.edof = (p-1)
        self.fdof = (p-1)*(p-1)//2
        self.cdof = self.fdof

        self.init_edge_to_dof()
        self.init_face_to_dof()
        self.init_cell_to_dof()
        self.init_orth_matrices()
        self.integralalg = self.space.integralalg
        self.integrator = self.integralalg.integrator

    def __str__(self):
        return "HuZhang Finite Element Space 2D!"

    def number_of_global_dofs(self) -> int:
        """
        Return the number of global degrees of freedom.
        """
        p = self.p
        GD = self.geo_dimension()
        tdim = self.tensor_dimension()

        mesh = self.mesh

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = tdim*NN

        if p > 1:
            edof = self.edof
            NE = mesh.number_of_edges()
            gdof += (tdim-1)*edof*NE # 边内部连续自由度的个数 
            E = mesh.number_of_edges_of_cells() # 单元边的个数
            gdof += NC*E*edof # 边内部不连续自由度的个数 

        if p > 2:
            fdof = self.fdof # 面内部自由度的个数
            if GD == 2:
                gdof += tdim*fdof*NC

        return gdof 

    def tensor_dimension(self):
        GD = self.GD
        return GD*(GD - 1)//2 + GD

    def number_of_local_dofs(self) -> int:
        """
        Return the number of local degrees of freedom.
        """
        ldof = self.dof.number_of_local_dofs()
        tdim = self.tensor_dimension()
        return ldof*tdim

    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.cell_to_dof()[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        return self.dof.face_to_dof()[index]

    def edge_to_dof(self, index=_S):
        return self.dof.edge_to_dof()[index]

    def init_cell_to_dof(self) -> TensorLike:
        """
        Constructing a mapping matrix from local degrees of freedom to global degrees of freedom

        Returns
        -------
        cell2dof : ndarray with shape (NC, ldof*tdim)
            NC: number of cells
            ldof: number of local degrees of freedom of the p-th order scalar space
            tdim: dimension of the symmetric tensor
        """
        pass

    def init_face_to_dof(self):
        """
        Constructing a mapping matrix from local degrees of freedom to global degrees of freedom
        Returns
        -------
        face2dof : ndarray with shape (NF, ldof,tdim)
            NF: number of faces
            ldof: number of local degrees of freedom of the p-th order scalar space
            tdim: dimension of the symmetric tensor
        """
        self.face2dof = self.edge_to_dof()[:]

    def init_edge_to_dof(self):
        """
        Constructing a mapping matrix from local degrees of freedom to global degrees of freedom
        Returns
        -------
        edge2dof : ndarray with shape (NE, ldof,tdim)
            NE: number of edges
            ldof: number of local degrees of freedom of the p-th order scalar space
            tdim: dimension of the symmetric tensor
        """
        pass

    def init_orth_matrices(self):
        """
        Constructing the orthogonal matrices for the p-th order Hu-Zhang finite element space
        Returns
        -------
        orth_matrices : ndarray with shape (NC, ldof, ldof)
            NC: number of cells
            ldof: number of local degrees of freedom of the p-th order scalar space
        """
        pass
    
    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def interpolation_points(self) -> TensorLike:
        return self.dof.interpolation_points()
    
    def dof_flags(self):
        """ 
        Classify the degrees of freedom in scalar space into edge internal degrees of freedom, face internal degrees of freedom (if in three-dimensional space), and other degrees of freedom

        Returns
        -------

        isOtherDof : ndarray, (ldof,) Other degrees of freedom except for those within edges and surfaces
        isEdgeDof : ndarray, (ldof, 3) or (ldof, 6) The degrees of freedom within each edge 
        isFaceDof : ndarray, (ldof, 4) The degrees of freedom within each face
        -------

        """
        pass

    def face_dof_flags(self):
        """
        Classify the degrees of freedom on face in scalar space into:
        Degree of freedom on the point
        The degree of freedom within the edge
        The degree of freedom within the face
        
        """
        p = self.p
        GD = self.geo_dimension()
        if GD == 2:
            return self.edge_dof_flags()
        else:
            raise ValueError('`dim` should be 2!')
    
    def edge_dof_flags(self):
        """
        Classify the degrees of freedom of basis functions on edges in scalar
        space into: degrees of freedom on points
        """ 
        p = self.p
        TD = 1
        multiIndex = self.mesh.multi_index_matrix(p, TD)#(ldof,2)
        isPointDof = (bm.sum(multiIndex == p, axis=-1) > 0)    
        isEdgeDof0  = ~isPointDof
        return isPointDof, isEdgeDof0    

