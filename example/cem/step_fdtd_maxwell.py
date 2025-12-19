from fealpy.cem.simulation.step_fdtd_maxwell import StepFDTDMaxwell
from fealpy.backend import backend_manager as bm
from fealpy.cem.mesh import YeeUniformMesher


Yee = YeeUniformMesher((0,1,0,1), nx=40, ny=40)
E = {
    'x': None, 
    'y': None,  
    'z': Yee.init_field_matrix("E", "z")  
}

H = {
    'x': Yee.init_field_matrix("H", "x"), 
    'y': Yee.init_field_matrix("H", "y"),  
    'z': None 
}
D = {
    'x': None,  
    'y': None,   
    'z': Yee.init_field_matrix("E", "z")  
}
B = {
    'x': Yee.init_field_matrix("H", "x"),  
    'y': Yee.init_field_matrix("H", "y"), 
    'z': None 
}

solver = StepFDTDMaxwell(Yee,R = 0.5, boundary='UPML', pml_width=8, eps=1.0, mu=1.0)
E, H, D, B  = solver.update(E, H, D, B)