from .config import *
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed

class MMesher:
    registered_param = {}
    def __init__(self,
                 mesh: Union[_U,list] ,
                 beta:Union[float,list],
                 vertices:Union[TensorLike,list],
                 r:Union[float,list] = 0.15,
                 is_multi_phy:bool = False,
                 config : Union[Config,list] = Config()):
        """
        @param mesh: mesh instance include TriangleMesh,TetrahedronMesh,
                                            QuadrangleMesh,HexahedronMesh,
                                            LagrangeTriangleMesh,LagrangeQuadrangleMesh
        @param uh: solution of the problem
        @param beta: parameter of the monitor function
        @param vertices: vertices of the domain
        """
        self.mesh = mesh
        self.beta = beta
        self.vertices = vertices
        self.r = r
        self.is_multi_phy = is_multi_phy
        self.config = config
        self.classes = {}
        self.preprocessed = False
        self.dim = 1

        self._check()
    
    def _check(self):
        if isinstance(self.mesh, list) and len(self.mesh) == 1:
            self.mesh = self.mesh[0]

        if isinstance(self.beta , list):
            self.dim = len(self.beta)
            kwargs = bm.context(self.mesh.node)
            self.beta = bm.array(self.beta, **kwargs)

        if (isinstance(self.beta ,(list,TensorLike)) and not isinstance(self.mesh , list) 
                                        and not self.is_multi_phy):
            from copy import deepcopy
            self.dim = len(self.beta)
            original_mesh = self.mesh  
            self.mesh = []  
            for _ in range(self.dim):
                new_mesh = deepcopy(original_mesh)
                new_mesh.node = deepcopy(original_mesh.node)
                self.mesh.append(new_mesh)

        if isinstance(self.mesh, list):
            for m in self.mesh:
                if not isinstance(m, _U.__args__):
                    raise TypeError(f"Each mesh in the list must be one of the types: \
                                    {', '.join([t.__name__ for t in _U.__args__])}")

        elif not isinstance(self.mesh, _U):
            raise TypeError(f"mesh must be one of the types: \
                            {', '.join([t.__name__ for t in _U.__args__])}")
        
        if not isinstance(self.beta, (float,int,list,TensorLike)):
            raise TypeError("beta must be a float or list")
        
        if not isinstance(self.vertices, (TensorLike,list)):
            raise TypeError("vertices must be tensor-like data or list")

        if isinstance(self.vertices, TensorLike) and self.dim > 1 and not self.is_multi_phy:
            self.vertices = [self.vertices] * self.dim

        if isinstance(self.r, float) and self.dim > 1 and not self.is_multi_phy:
            self.r = [self.r] * self.dim
        
        if self.is_multi_phy:
            self.config.monitor = 'mp_arc_length'
            self.config.int_meth = 'mp_comass'

    def register(self, parameters: dict):
        """
        Register or update parameters.
        @param parameters: A dictionary containing parameter names and their values.
        """
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary")

        valid_keys = vars(self.config).keys()  # Retrieve all parameter names from the Config object
        for key in parameters.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter name: {key}. Allowed parameters: {list(valid_keys)}")
        # if instance is not initialized, register parameters
        self.registered_param.update(parameters)
        if hasattr(self, 'instance_list'):
            for instance in self.instance_list:
                for key, value in parameters.items():
                    setattr(self.config, key, value)  
                    self._update_recursive(instance, key, value)
        elif hasattr(self, 'instance'):
            for key, value in parameters.items():
                setattr(self.config, key, value)  
                self._update_recursive(self.instance, key, value)
        self._initialize_param()
        self._check()

    def _update_recursive(self,obj, key, value):
        """
        Recursively update the parameter in the object and its parent classes.
        @param obj: The current object to update.
        @param key: The name of the parameter to update.
        @param value: The new value to set for the parameter.
        """
        if hasattr(obj, key):
            setattr(obj, key, value)
        for base_class in obj.__class__.__bases__:  # Iterate through the parent classes
            if hasattr(base_class, key):
                setattr(base_class, key, value)

    def _initialize_param(self):
        """
        @brief initialize parameters, prefer to use the parameters registered by the user
        """
        final_param = {}
        for key, default_value in vars(self.config).items():
            value = self.registered_param.get(key, default_value)
            setattr(self.config, key, value)
            final_param[key] = value
        return final_param
    
    def _load_class(self, module_name: str, class_name: str):
        """
        @brief load class from module
        """
        try:
            module = __import__(f"fealpy.mmesh.{module_name}", fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Failed to import module 'fealpy.mmesh.{module_name}': {e}")
        except AttributeError as e:
            raise AttributeError(f"Failed to find class '{class_name}' in module 'fealpy.mmesh.{module_name}': {e}")
        
    def initialize(self):
        """
        @brief according to the config file, load the active_method
        """
        if not isinstance(self.beta, (list,TensorLike)) or self.is_multi_phy:
            class_name  = self.config.active_method
            module_name = f"{class_name.lower()}"
            class_ = self._load_class(module_name, class_name)
            self.instance = class_(self.mesh,self.beta,self.vertices,self.r,self.config)
            self._catch()
            print(f"Initializing {class_name} !")
        else:
            if self.config.parallel_mode == "thread":
                self.executor = ThreadPoolExecutor()
            elif self.config.parallel_mode == "process":
                self.executor = ProcessPoolExecutor()
            self.instance_list = []
            self.process_list = []
            class_name  = self.config.active_method
            module_name = f"{class_name.lower()}"
            class_ = self._load_class(module_name, class_name)  
            for i in range(self.dim):
                instance = class_(self.mesh[i],self.beta[i],self.vertices[i],self.r[i],self.config)
                self.instance_list.append(instance)
                self._catch(instance)
            print(f"Initializing {class_name} !")
    
    def _catch(self,instance = None):
        """
        @brief catch the process function
        """
        if self.is_multi_phy:
            if self.config.is_pre and not self.preprocessed:
                self.process = self.instance.mp_preprocessor
                self.preprocessed = True
            else:
                self.process = self.instance.mp_mesh_redistributor
        else:        
            if isinstance(self.mesh, list):
                if self.config.is_pre and not self.preprocessed:
                    self.process_list.append(instance.preprocessor)
                    self.preprocessed = True
                else:
                    self.process_list.append(instance.mesh_redistributor)
            else:
                if self.config.is_pre and not self.preprocessed:
                    self.process = self._preprocessor_wrapper
                    self.preprocessed = True
                else:
                    self.process = self.instance.mesh_redistributor
        
    def _preprocessor_wrapper(self):
        """
        Return a reference to the preprocessor method.
        """
        fun_solver = self.config.fun_solver
        self.instance.preprocessor(fun_solver)
        
    def run(self,uh : TensorLike):
        """
        @brief run the active_method, applicable to scalar cases
        """
        return self._run_instance(self.instance, 
                                  self.process, 
                                  uh, 
                                  self.config.active_method)
    
    def run_multi(self,uh_list: list):
        """
        @brief run the active_method, applicable to multi-mesh cases
        """
        # none parallel mode
        if self.config.parallel_mode == "none":
            results = []
            for i in range(self.dim):
                result = self._run_instance(self.instance_list[i],
                                            self.process_list[i],
                                            uh_list[i], 
                                            self.config.active_method)
                results.append(result)
            mesh_list, uh_list = zip(*results)

        # thread mode or process mode
        elif self.config.parallel_mode in ["thread", "process"]:
            futures = [
                self.executor.submit(
                    self._run_instance,
                    self.instance_list[i],
                    self.process_list[i],
                    uh_list[i],
                    self.config.active_method
                )
                for i in range(self.dim)
            ]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            mesh_list, uh_list = zip(*results)
        return list(mesh_list), list(uh_list)

    def run_mp(self,uh: TensorLike):
        """
        @brief run the active_method, applicable to multi-physics cases
        """
        return self._run_instance(self.instance,
                                  self.process, 
                                  uh, 
                                  self.config.active_method,
                                  self.dim)
    
    @staticmethod
    def _run_instance(instance, process, uh, active_method,dim = 1):
        """
        Run the process for a specific instance.
        """
        # Set the uh for the current instance
        instance.uh = uh
        instance.dim = dim
        # Run the process
        print(f"Running {active_method} for instance!")
        process()
        # Return the updated uh and mesh
        return instance.mesh, instance.uh
    
    def shutdown(self):
        """
        Shutdown the executor (thread pool or process pool).
        """
        if self.executor:
            self.executor.shutdown()

    def restart(self,uh):
        """
        @brief update the solution of the problem
        """
        instance = self.instance
        instance.uh = uh
        if self.is_multi_phy:
            self.process = self.instance.mp_mesh_redistributor
        else:
            self.process = self.instance.mesh_redistributor
        

    def show_mesh(self,ax,scat_node = True , scat_index = slice(None)):
        instance = self.instance
        mesh = instance.mesh
        if instance.mesh_type in ["LagrangeTriangleMesh", "LagrangeQuadrangleMesh"]:
            from .tool import high_order_meshploter
            high_order_meshploter(ax,mesh,scat_node = scat_node , scat_index = scat_index)
        else:
            mesh.add_plot(ax)
        # ax.clear()

    def show_function(self,ax,uh,scat_node = True , index = slice(None)):
        from .tool import high_order_meshploter,linear_surfploter
        instance = self.instance
        mesh = instance.mesh
        if instance.mesh_type in ["LagrangeTriangleMesh", "LagrangeQuadrangleMesh"]:    
            high_order_meshploter(ax,mesh,uh,model='surface',
                                  scat_node = scat_node, scat_index = index)
        else:
            linear_surfploter(ax,mesh,uh,
                              scat_node = scat_node, scat_index = index)
        # ax.clear()