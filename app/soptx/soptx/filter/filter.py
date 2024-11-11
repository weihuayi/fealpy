from typing import Optional
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .types import FilterProperties, FilterType

class TopologyFilter:
    """拓扑优化过滤器
    
    实现各种过滤方案的统一接口，用于处理密度场和灵敏度。
    """
    
    def __init__(self, filter_props: Optional[FilterProperties] = None):
        """初始化过滤器
        
        Args:
            filter_props: 过滤器属性，如果为None则不执行过滤
        """
        self.props = filter_props
    
    def filter_density(self, rho: TensorLike) -> TensorLike:
        """过滤密度场
        
        Args:
            rho: 原始密度场
            
        Returns:
            TensorLike: 过滤后的密度场
        """
        if self.props is None or self.props.H is None:
            return rho
            
        if self.props.filter_type == FilterType.DENSITY:
            cell_measure = self.props.mesh.entity_measure('cell')
            H = self.props.H
            return H.matmul(rho * cell_measure) / H.matmul(cell_measure)
            
        return rho
    
    def filter_sensitivity(
        self,
        dce: TensorLike,
        rho: Optional[TensorLike] = None,
        beta: Optional[float] = None,
        rho_tilde: Optional[TensorLike] = None
    ) -> TensorLike:
        """过滤灵敏度
        
        Args:
            dce: 原始灵敏度
            rho: 密度场（用于灵敏度过滤）
            beta: Heaviside投影参数
            rho_tilde: 过滤后的密度场
            
        Returns:
            TensorLike: 过滤后的灵敏度
            
        Raises:
            ValueError: 当所需参数缺失时
        """
        if self.props is None or self.props.H is None:
            return dce
            
        H = self.props.H
        cell_measure = self.props.mesh.entity_measure('cell')
        
        if self.props.filter_type == FilterType.SENSITIVITY:
            if rho is None:
                raise ValueError("Density field required for sensitivity filter")
            rho_dce = bm.einsum('c, c -> c', rho, dce)
            filtered_dce = H.matmul(rho_dce)
            return filtered_dce / self.props.Hs / bm.maximum(0.001, rho)
            
        elif self.props.filter_type == FilterType.HEAVISIDE:
            if beta is None or rho_tilde is None:
                raise ValueError("Heaviside projection filter requires beta and rho_tilde")
            dxe = beta * bm.exp(-beta * rho_tilde) + bm.exp(-beta)
            return H.matmul(dce * dxe * cell_measure) / H.matmul(cell_measure)
            
        return dce