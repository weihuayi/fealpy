from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.functionspace.tensor_space import TensorFunctionSpace
from fealpy.typing import TensorLike

def calculate_ke0(material_properties: LinearElasticMaterial, 
                tensor_space: TensorFunctionSpace) -> TensorLike:
    """
    Calculate the element stiffness matrix assuming E=1.

    Args:
        tensor_space: TensorFunctionSpace object for the computational space.

    Returns:
        TensorLike: The element stiffness matrix ke0.
    """
    base_material = LinearElasticMaterial(name='base_material', 
                                        elastic_modulus=material_properties.E, 
                                        poisson_ratio=material_properties.nu, 
                                        hypo=material_properties.hypo)
    
    integrator = LinearElasticIntegrator(material=base_material, q=tensor_space.p + 3)
    
    ke0 = integrator.assembly(space=tensor_space)

    return ke0
