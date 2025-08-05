import inspect

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.fem import LinearForm, BilinearForm
from fealpy.fem import (ScalarMassIntegrator, FluidBoundaryFrictionIntegrator, 
                     ViscousWorkIntegrator, SourceIntegrator, GradSourceIntegrator, 
                     BoundaryFaceSourceIntegrator, ScalarDiffusionIntegrator)
from fealpy.functionspace import Function
from fealpy.decorator import barycentric
from fealpy.fem import DirichletBC

from ..project_method import ProjectionMethod

class IPCS(ProjectionMethod):
    """IPCS分裂投影法"""
    
    def predict_velocity(self, u0:TensorLike, p0:TensorLike, return_form:bool=True,BC=None): 
        Bform = self.predict_velocity_BForm()
        Lform = self.predict_velocity_LForm()
        self.predict_velocity_update(u0, p0)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if BC is not None:
                A, b = BC.apply(A, b)
            return A, b 
        else:
            return Bform, Lform 
    
    def pressure(self, us:TensorLike, p0:TensorLike,  return_form=True, BC=None):
        Bform = self.pressure_BForm()
        Lform = self.pressure_LForm()
        self.pressure_update(us, p0)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if self.equation.pressure_neumann == True:
                A, b = self.lagrange_multiplier(A, b, 0)
            else:
                A, b = BC.apply(A, b)
            return A, b
        else:
            return Bform, Lform 
    
    def correct_velocity(self, us:TensorLike, p0:TensorLike, p1:TensorLike, 
                         return_form=True, BC=None):
        """速度校正"""
        Bform = self.correct_velocity_BForm()
        Lform = self.correct_velocity_LForm()
        self.correct_velocity_update(us, p0, p1)
        if return_form is False:
            A = Bform.assembly()
            b = Lform.assembly()
            if BC is not None:
                A, b = BC.apply(A, b)
            return A, b
        else:
            return Bform, Lform 

    def predict_velocity_BForm(self):
        """预测速度左端项"""
        uspace = self.uspace
        q = self.q 

        Bform = BilinearForm(uspace)
        self.predict_BM = ScalarMassIntegrator(q=q)  
        Bform.add_integrator(self.predict_BM)
        
        if self.equation.pressure_neumann == False:
            self.predict_BF = FluidBoundaryFrictionIntegrator(q=q, threshold=self.threshold)
            Bform.add_integrator(self.predict_BF)
        
        if self.equation.constitutive.value == 1:
            self.predict_BVW = ScalarDiffusionIntegrator(q=q)
            Bform.add_integrator(self.predict_BVW)
        elif self.equation.constitutive.value == 2:
            self.predict_BVW = ViscousWorkIntegrator(q=q)
            Bform.add_integrator(self.predict_BVW)
        else :
            raise ValueError(f"未知的粘性模型")
        return Bform
    
    def predict_velocity_LForm(self):
        """预测速度右端项"""
        uspace = self.uspace
        q = self.q
        
        Lform = LinearForm(uspace) 
        self.predict_LS = SourceIntegrator(q=q)
        self.predict_LS_f = SourceIntegrator(q=q)
        self.predict_LGS = GradSourceIntegrator(q=q)
        
        Lform.add_integrator(self.predict_LS)
        Lform.add_integrator(self.predict_LGS)
        Lform.add_integrator(self.predict_LS_f)
        if self.equation.pressure_neumann == False:
            self.predict_LBFS = BoundaryFaceSourceIntegrator(q=q, threshold=self.threshold)
            Lform.add_integrator(self.predict_LBFS)
        return Lform
    
    def predict_velocity_update(self, u0, p0): 
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative 
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        self.predict_BM.coef = ctd/dt
        self.predict_BVW.coef = cv
        
        @barycentric
        def LS_coef(bcs, index):
            masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = 1/dt*masscoef*u0(bcs, index)
            ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            result -= ccoef * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result

        @barycentric
        def LGS_coef(bcs, index):
            I = bm.eye(self.mesh.GD)
            result = bm.repeat(p0(bcs,index)[...,bm.newaxis], self.mesh.GD, axis=-1)
            result = bm.expand_dims(result, axis=-1) * I
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        
        
        @barycentric
        def LBFS_coef(bcs, index):
            result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), self.mesh.face_unit_normal(index=index))
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        
        self.predict_LS_f.source = cbf
        self.predict_LS.source = LS_coef
        self.predict_LGS.source = LGS_coef
        if self.equation.pressure_neumann == False:
            self.predict_BF.coef = -cv
            self.predict_LBFS.source = LBFS_coef


    def pressure_BForm(self):
        """压力泊松方程左端项"""
        pspace = self.pspace
        q = self.q
         
        Bform = BilinearForm(pspace)
        self.pressure_BD = ScalarDiffusionIntegrator(q=q)
        Bform.add_integrator(self.pressure_BD) 
        return Bform
    
    def pressure_LForm(self):
        """压力泊松方程右端项"""
        pspace = self.pspace
        q = self.q
         
        Lform = LinearForm(pspace)
        self.pressure_LS = SourceIntegrator(q=q)
        self.pressure_LGS = GradSourceIntegrator(q=q)
        
        Lform.add_integrator(self.pressure_LS)
        Lform.add_integrator(self.pressure_LGS)
        return Lform
    
    def pressure_update(self, us, p0): 
        equation = self.equation
        dt = self.dt
        pc = equation.coef_pressure
        ctd = equation.coef_time_derivative 
        
        self.pressure_BD.coef = pc

        @barycentric
        def LS_coef(bcs, index=None):
            result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
            result *= ctd(bcs, index) if callable(ctd) else ctd
            return result
        self.pressure_LS.source = LS_coef
        
        @barycentric
        def LGS_coef(bcs, index=None):
            result = p0.grad_value(bcs, index)
            result *= pc(bcs, index) if callable(pc) else pc
            return result
        self.pressure_LGS.source = LGS_coef
    
    def correct_velocity_BForm(self):
        """速度校正左端项"""
        uspace = self.uspace
        q = self.q
        
        Bform = BilinearForm(uspace)
        self.correct_BM = ScalarMassIntegrator(q=q)
        Bform.add_integrator(self.correct_BM)
        return Bform
    
    def correct_velocity_LForm(self):
        """速度校正右端项"""
        uspace = self.uspace
        q = self.q
        
        Lform = LinearForm(uspace)
        self.correct_LS = SourceIntegrator(q=q)
        Lform.add_integrator(self.correct_LS)
        return Lform

    def correct_velocity_update(self, us, p0, p1):
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative
        cp = equation.coef_pressure


        self.correct_BM.coef = ctd
        @barycentric
        def BM_coef(bcs, index):
            masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = masscoef * us(bcs, index)
            result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
            return result
        self.correct_LS.source = BM_coef














