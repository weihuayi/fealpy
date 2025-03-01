
from .base import MaterialInterpolation
# TODO 删除
from .elastic import (ElasticMaterialProperties, 
                      ElasticMaterialInstance, 
                      ElasticMaterialConfig)
from .linear_elastic_material import (ElasticMaterialInstance, 
                                      ElasticMaterialConfig)
from .interpolation_scheme import (SIMPInterpolation, 
                                   RAMPInterpolation)

__all__ = [
    'MaterialInterpolation',
    'ElasticMaterialInstance',
    'ElasticMaterialConfig',
    'SIMPInterpolation',
    'RAMPInterpolation'
]
