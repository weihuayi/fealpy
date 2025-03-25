from .config import *
import importlib

class MMesher:
    def __init__(self,config: Config):
        self.config = config
        self.classes = {}
        self.preprocessed = False
    
    def initialize(self):
        """
        @brief according to the config file, load the active_method
        """
        class_name  = self.config.active_method
        module_name = f"{class_name.lower()}"
        try:
            module = __import__(f"fealpy.mmesh.{module_name}", fromlist=[class_name])
            class_ = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_name} from {module_name}: {e}")

        # 获取类
        class_ = getattr(module, class_name)
        self.instance = class_(self.config)
        self._catch()
        print(f"Initializing {class_name} !")
    
    def _catch(self):
        """
        @brief catch the process function
        """
        if self.config.is_pre and not self.preprocessed:
            self.process = self._preprocessor_wrapper
            self.preprocessed = True
        else:
            self.process = self.instance.mesh_redistributor

    def _preprocessor_wrapper(self):
        fun_solver = self.config.fun_solver
        self.instance.preprocessor(fun_solver)

    def run(self):
        """
        @brief run the active_method
        """
        instance = self.instance
        print(f"Running {self.config.active_method} !")
        self.process()
        return instance.mesh, instance.uh
    
    def restart(self,uh):
        """
        @brief update the solution of the problem
        """
        instance = self.instance
        instance.uh = uh
        self.process = self.instance.mesh_redistributor

    def show_mesh(self,scat_node = True , scat_index = slice(None)):
        instance = self.instance
        mesh = instance.mesh
        if instance.mesh_type in ["LagrangeTriangleMesh", "LagrangeQuadrangleMesh"]:
            from .tool import high_order_meshploter
            high_order_meshploter(mesh,scat_node = scat_node , scat_index = scat_index)
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            if instance.TD == 2:
                ax = fig.gca()
            else:
                ax = fig.add_subplot(111, projection='3d')
            mesh.add_plot(ax)
            plt.show()

    def show_function(self,uh,scat_node = True , index = slice(None)):
        from .tool import high_order_meshploter,linear_surfploter
        instance = self.instance
        mesh = instance.mesh
        if instance.mesh_type in ["LagrangeTriangleMesh", "LagrangeQuadrangleMesh"]:    
            high_order_meshploter(mesh,uh,model='surface',
                                  scat_node = scat_node, scat_index = index)
        else:
            linear_surfploter(mesh,uh,
                              scat_node = scat_node, scat_index = index)