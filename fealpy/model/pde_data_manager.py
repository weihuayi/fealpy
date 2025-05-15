import importlib
import os
import sys

class PDEDataManager:
    """
    PDEDataManager manages PDE model types and examples for FEALPy framework.
    This class provides a unified interface to list supported PDE types,
    browse available PDE model examples under a type, and instantiate a specific model.
    It helps both new and advanced users to interact with FEALPy's PDE library.

    Parameters
        pde_type : str, optional
            The category of PDE models, including 'poisson', 'parabolic', 'wave', 'elliptic','hyperbolic'.
            If not provided, only type listing is supported.

    Attributes
        pde_type : str or None
            The current PDE type set by the user. Determines which example table is loaded.
        module_path : str or None
            The Python module path corresponding to the selected PDE type.
        data_table : dict
            Dictionary mapping example keys to tuples of (file_name, class_name).

    Methods
        show_types()
            Class method to print all available PDE types.
        show_examples()
            Print all available examples for the current PDE type.
        get_example(key)
            Instantiate and return a PDE model based on its example key.

    Examples
        >>> PDEDataManager.show_types()
        >>> manager = PDEDataManager('poisson')
        >>> manager.show_examples()
        >>> pde = manager.get_example('coscos')
    """

    _registry = {
        "poisson": "fealpy.model.poisson",
        "elliptic": "fealpy.model.elliptic",
        "parabolic": "fealpy.model.parabolic",
        "wave": "fealpy.model.wave",
        "hyperbolic":"fealpy.model.hyperbolic"
    }

    def __init__(self, pde_type: str = None):
        """
        Initialize the manager and optionally load a PDE type's example table.

        Parameters
            pde_type : str, optional
                The PDE category to load (e.g., 'poisson', 'wave', 'parabolic', 'elliptic'). If not set,
                only show_types() is available.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        self.pde_type = pde_type
        self.module_path = None
        self.data_table = {}

        if pde_type is not None:
            if pde_type not in self._registry:
                raise ValueError(f"[Error] Unknown PDE type: '{pde_type}'")
            self.module_path = self._registry[pde_type]
            module = importlib.import_module(self.module_path)
            self.data_table = getattr(module, "DATA_TABLE", {})

    @classmethod
    def show_types(cls):
        """
        Print all supported PDE types.

        Returns
            None

        Examples
            >>> PDEDataManager.show_types()
        """
        print("Available PDE types:")
        for key in cls._registry:
            print(f" - {key}")

    def show_examples(self):
        """
        Print all available PDE models under the current type.

        Raises
            RuntimeError
                If no PDE type has been set during initialization.

        Returns
            None

        Examples
            >>> manager = PDEDataManager("poisson")
            >>> manager.show_examples()
        """
        if not self.pde_type or not self.data_table:
            raise RuntimeError("PDE type not set. Please initialize with a valid pde_type.")
        print(f"Available examples for PDE type '{self.pde_type}':")
        print("\n examples name: (file_name, class_name)")
        for key, (file_name, class_name) in self.data_table.items():
            print(f" - {key}: ({file_name}, {class_name})")
        print(f"\nExample usage:\n   pde = PDEDataManager('{self.pde_type}').get_example('example name')")

    def get_example(self, key: str = None):
        """
        Instantiate and return a PDE model object based on example key.

        Parameters
            key : str, optional
                The example name key from the current PDE type (e.g., 'coscos').

        Returns
            instance : object
                An instance of the selected PDE model class.

        Examples
            >>> pde = PDEDataManager("poisson").get_example("coscos")
        """
        if not self.pde_type or not self.data_table:
            raise RuntimeError("PDE type not set. Please initialize with a valid pde_type.")

        if key is None:
            raise ValueError(
                f"[Error] No model name provided.\n"
                f"Please pass a valid example key, like:\n"
                f"    PDEDataManager('{self.pde_type}').get_example('coscos')\n"
                f"Use .show_examples() to see all available models."
            )

        if key not in self.data_table:
            raise ValueError(f"[Error] Unknown key: '{key}'. Use .show_examples() to view valid keys.")

        file_name, class_name = self.data_table[key]
        submodule = importlib.import_module(f"{self.module_path}.{file_name}")
        cls = getattr(submodule, class_name)
        return cls()
