from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from fealpy.experimental.typing import TensorLike

def calculate_KE(material_properties, tensor_space) -> TensorLike:
    """
    Calculate the element stiffness matrix assuming E=1.

    Args:
        tensor_space: TensorFunctionSpace object for the computational space.

    Returns:
        TensorLike: The element stiffness matrix KE.
    """
    base_material = LinearElasticMaterial(name='base_material', 
                                        elastic_modulus=material_properties.E0, 
                                        poisson_ratio=material_properties.nu, 
                                        hypo=material_properties.hypo)
    
    integrator = LinearElasticIntegrator(material=base_material, q=tensor_space.p + 3)
    
    KE = integrator.assembly(space=tensor_space)

    return KE
