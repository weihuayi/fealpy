from fealpy.backend import backend_manager as bm
from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell


pde = PointSourceMaxwell(eps=2.5, mu=1.0)  

tag0 = pde.add_source(comp='Ez', waveform='gaussian_enveloped_sine',
                      waveform_params={'freq':1e9, 't0':1.0, 'tau':0.2},
                      amplitude=1.0, spread=0, injection='soft')

tag1 = pde.add_source(position=(0.2, 0.3, 0.5), comp='Ex',
                      waveform='sinusoid', waveform_params={'freq':2e9},
                      amplitude=0.5)

configs = pde.get_source_config()
print(configs)

print(pde.summary())
