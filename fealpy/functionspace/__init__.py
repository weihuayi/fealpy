
from .space import FunctionSpace
from .function import Function

from .dofs import LinearMeshCFEDof

from .lagrange_fe_space import LagrangeFESpace
from .tensor_space import TensorFunctionSpace
from .cm_conforming_fe_space_2d import CmConformingFESpace2d
from .cm_conforming_fe_space_3d import CmConformingFESpace3d
from .bernstein_fe_space import BernsteinFESpace

from .first_nedelec_fe_space import FirstNedelecFESpace
from .first_nedelec_fe_space_2d import FirstNedelecFESpace2d
from .first_nedelec_fe_space_3d import FirstNedelecFESpace3d

from .second_nedelec_fe_space import SecondNedelecFESpace
from .second_nedelec_fe_space_2d import SecondNedelecFESpace2d
from .second_nedelec_fe_space_3d import SecondNedelecFESpace3d

from .raviart_thomas_fe_space import RaviartThomasFESpace
from .raviart_thomas_fe_space_2d import  RaviartThomasFESpace2d
from .raviart_thomas_fe_space_3d import  RaviartThomasFESpace3d

from .parametric_lagrange_fe_space import ParametricLagrangeFESpace

from .huzhang_fe_space_2d import HuZhangFESpace2d
from .huzhang_fe_space_3d import HuZhangFESpace3d

from .brezzi_douglas_marini_fe_space import BrezziDouglasMariniFESpace
from .brezzi_douglas_marini_fe_space_2d import BrezziDouglasMariniFESpace2d
from .brezzi_douglas_marini_fe_space_3d import BrezziDouglasMariniFESpace3d


from .interior_penalty_fe_space_2d import InteriorPenaltyFESpace2d

## VESpace
from .scaled_monomial_space_2d import ScaledMonomialSpace2d
from .conforming_scalar_ve_space_2d import ConformingScalarVESpace2d
from .non_conforming_scalar_ve_space_2d import NonConformingScalarVESpace2d



def functionspace(mesh, space_type, shape=None):
    if space_type[0] == 'Lagrange':
        scalar_space = LagrangeFESpace(mesh, space_type[1])
        if shape is not None:
            return TensorFunctionSpace(scalar_space, shape)
        else:
            return scalar_space



