import importlib
import os
import sys


class ModelManager:
    """DataManager manages model types and examples for FEALPy framework.

    This class provides a unified interface to list supported types,
    browse available model examples under a type, and instantiate a specific model.
    It helps both new and advanced users to interact with FEALPy.

    Parameters:
        model_type(str, optional): The category of models, such as 'poisson',
            'parabolic', 'wave', 'elliptic','hyperbolic'.
            If not provided, only type listing is supported.

    Attributes:
        model_type(str or None): The current model type set by the user.
            Determines which example table is loaded.
        module_path(str or None): The Python module path corresponding to the
            selected type.
        data_table(dict): Dictionary mapping example keys to tuples of
            (file_name, class_name).

    Methods:
        show_types()
            Class method to print all available types.
        show_examples()
            Print all available examples for the current type.
        get_example(key)
            Instantiate and return a model based on its example key.

    Examples:
        >>> ModelManager.show_types()
        >>> manager = ModelManager('poisson')
        >>> manager.show_examples()
        >>> model = manager.get_example('coscos')
    """

    _registry = {}

    def __init__(self, model_type: str):
        """
        Initialize the manager and optionally load a type table.

        Parameters:
            model_type(str): The model category to load (e.g., 'poisson', 'wave',
                'parabolic', 'elliptic', 'hyperbolic', 'helmholtz', 'curlcurl').
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        self.model_type = model_type
        self.module_path = None
        self.data_table = {}

        if model_type not in self._registry:
            raise ValueError(f"[Error] Unknown type: '{model_type}'")

        self.module_path = self._registry[model_type]
        module = importlib.import_module(self.module_path)
        self.data_table = getattr(module, "DATA_TABLE", {})

    @classmethod
    def show_types(cls) -> None:
        """
        Print all supported model types.

        Returns:
            None

        Examples:
            >>> ModelManager.show_types()
        """
        print("Available Model types:")
        for key in cls._registry:
            print(f" - {key}")

    def show_examples(self) -> None:
        """
        Print all available models under the current type.

        Raise:
            RuntimeError: If no Model type has been set during initialization.

        Returns:
            None

        Examples:
            >>> manager = ModelManager("poisson")
            >>> manager.show_examples()
        """
        print(f"Available examples for type '{self.model_type}':")
        print("\n examples name: (file_name, class_name)")
        for key, (file_name, class_name) in self.data_table.items():
            print(f" - {key}: ({file_name}, {class_name})")
        print(f"\nExample usage:\n   model = ModelManager('{self.model_type}').get_example('example name')")

    def get_example(self, key: int, **options):
        """Instantiate and return a model object based on example key.

        Parameters:
            key(int): The example name key from the current model type.

        Returns:
            instance(object): An instance of the selected model class.

        Examples:
            >>> model = ModelManager("poisson").get_example(1)
        """
        if key not in self.data_table:
            raise ValueError(f"[Error] Unknown key: '{key}'. Use .show_examples() to view valid keys.")

        file_name, class_name = self.data_table[key]
        submodule = importlib.import_module(f"{self.module_path}.{file_name}")
        cls = getattr(submodule, class_name)
        if options == {}:
            return cls()
        else:
            return cls(options)


class PDEModelManager(ModelManager):
    _registry = {
        "poisson": "fealpy.model.poisson",
        "diffusion": "fealpy.model.diffusion",
        "diffusion_convection": "fealpy.model.diffusion_convection",
        "diffusion_convection_reaction": "fealpy.model.diffusion_convection_reaction",
        "diffusion_reaction": "fealpy.model.diffusion_reaction",
        "parabolic": "fealpy.model.parabolic",
        "wave": "fealpy.model.wave",
        "hyperbolic":"fealpy.model.hyperbolic",
        "darcyforchheimer":"fealpy.model.darcyforchheimer",
        "interface_poisson":"fealpy.model.interface_poisson",
        "ion_flow":"fealpy.model.ion_flow",
        "nonlinear":"fealpy.model.nonlinear",
        "linear_elasticity": "fealpy.model.linear_elasticity",
        "quasilinear_elliptic": "fealpy.model.quasilinear_elliptic",
        "polyharmonic": "fealpy.model.polyharmonic",
        "stokes": "fealpy.model.stokes",
        "linear_elasticity": "fealpy.model.linear_elasticity",
        "allen_cahn": "fealpy.model.allen_cahn",
        "optimal_control": "fealpy.model.optimal_control",
        "helmholtz": "fealpy.model.helmholtz",
        "surface_poisson": "fealpy.model.surface_poisson",
        "curlcurl":"fealpy.model.curlcurl"
    }