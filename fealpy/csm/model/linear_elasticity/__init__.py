
"""
DATA_TABLE is a registry, when adding new PDE models, 
follow the existing examples to register them in the registry.
"""
DATA_TABLE = {
    # example name: (file_name, class_name)
    1: ("linear_elasticity_data_3d", "LinearElasticityData3D"),
    2: ("gear_box_model", "GearBoxModel"),
}
