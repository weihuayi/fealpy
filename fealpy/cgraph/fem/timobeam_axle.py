from fealpy.backend import bm
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TimobeamAxleEquation"]

class TimobeamAxle(CNodeType):
    TITLE: str = "TimobeamAxleEquation"
    PATH: str = "fem.presets"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE),
        PortConf("q", DataType.INT, default=3, min_val=1, max_val=17),
        PortConf("external_load", DataType.TENSOR),
        PortConf("beam_E", DataType.FLOAT),
        PortConf("beam_nu", DataType.FLOAT),
        PortConf("beam_mu", DataType.FLOAT),
        PortConf("axle_E", DataType.FLOAT),
        PortConf("axle_nu", DataType.FLOAT),
        PortConf("axle_mu", DataType.FLOAT)
    ]
    OUTPUT_SLOTS = [
        PortConf("operator", DataType.LINOPS),
        PortConf("external_load", DataType.TENSOR)
    ]
    
    @staticmethod
    def run(pde, mesh, space, beam_E, beam_nu, axle_E, axle_nu, external_load):
        from fealpy.csm.material import (
            TimoshenkoBeamMaterial,
            AxleMaterial
        )
        from fealpy.csm.fem import (
           TimoshenkoBeamIntegrator,
           AxleIntegrator
        )
        
        from ...fem import (
            BilinearForm,
        )
        
        Timo = TimoshenkoBeamMaterial(name="timobeam",
                                    elastic_modulus=beam_E,
                                    poisson_ratio=beam_nu)
                
        Axle = AxleMaterial(name="axle",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        
        bform_beam = BilinearForm(space)
        bform_beam.add_integrator(TimoshenkoBeamIntegrator(space, Timo, 
                                        index=bm.arange(0, mesh.number_of_cells()-10)))
        beam_K = bform_beam.assembly(format='csr')

        bform_axle = BilinearForm(space)
        bform_axle.add_integrator(AxleIntegrator(space, Axle, 
                                        index=bm.arange(mesh.number_of_cells()-10, mesh.number_of_cells())))
        axle_K = bform_axle.assembly(format='csr')

        K = beam_K + axle_K
        F = external_load

        return K, F