
import os
import sys
import shutil
import logging
from setuptools import Extension
from abc import ABC, abstractmethod
import platform as plat

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseInstaller(ABC):
    """
    Abstract base class for platform-specific installers. Each platform must implement
    the `install_libraries`, `install_headers`, and `build` methods.
    """

    def __init__(self, dep, ext_modules):
        """
        Initialize the installer with dependencies and extension modules.

        Parameters:
            dep (dict): Dependency information (e.g., install_prefix).
            ext_modules (list): List of extension modules to be updated.
        """
        self.dep = dep
        self.ext_modules = ext_modules

    @abstractmethod
    def update(self):
        """
        Update the list of extension modules for the platform.
        """
        pass

    @abstractmethod
    def install_libraries(self, datatype, platform):
        """
        Install libraries for the specified datatype and platform.
        """
        pass

    @abstractmethod
    def install_headers(self, datatype):
        """
        Install headers for the specified datatype.
        """
        pass

    @abstractmethod
    def build(self):
        """
        Build and install the Pangulu libraries, headers, and generate Makefile for compilation.
        """
        pass


class LinuxInstaller(BaseInstaller):
    def update(self):
        """
        Updates the list of extension modules for Linux platform.
        """
        logging.info("Updating extension modules for Linux...")
        self.ext_modules.append(
            Extension(
                'fealpy.solver.pangulu._pangulu_r64_cpu',
                sources=['fealpy/solver/pangulu/_pangulu_r64.pyx'],
                libraries=['pangulu_r64_cpu', 'metis', 'openblas'],
                library_dirs=[os.path.expanduser('~/.local/lib')],
                include_dirs=[os.path.expanduser('~/.local/include')],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']
            ),
        )


    def install_libraries(self, datatype, platform):
        """
        Install libraries by moving dynamic and static libraries to the appropriate install directory.
        
        Parameters:
            datatype (str): The data type, should be 'R64' or 'R32'.
            platform (str): The platform, should be 'cpu' or 'gpu'.
        """
        pass


    def install_headers(self, datatype):
        """
        Install the appropriate headers for the specified datatype (R64 or R32).

        Parameters:
            datatype (str): The data type, should be 'R64' or 'R32'.
        """
        pass

    def build(self):
        """
        Build and install the Pangulu libraries, headers, and generate Makefile for compilation on Linux platform.
        """
        pass


class WindowsInstaller(BaseInstaller):
    def update(self):
        logging.info("Updating extension modules for Windows...")
        self.ext_modules.append(
            Extension(
                'fealpy.solver.pangulu._pangulu_r64_cpu',
                sources=['fealpy/solver/pangulu/_pangulu_r64.pyx'],
                libraries=['pangulu_r64_cpu', 'metis', 'openblas'],
                library_dirs=[os.path.expanduser('~/.local/lib')],
                include_dirs=[os.path.expanduser('~/.local/include')],
                extra_compile_args=['/openmp'],
                extra_link_args=['/openmp']
            ),
        )

    def install_libraries(self, datatype, platform):
        logging.info(f"Installing libraries for {datatype} on {platform} (Windows)...")
        super().install_libraries(datatype, platform)

    def install_headers(self, datatype):
        logging.info(f"Installing headers for {datatype} (Windows)...")
        super().install_headers(datatype)

    def build(self):
        logging.info("Building Pangulu libraries for Windows...")
        super().build()


class MacInstaller(BaseInstaller):
    def update(self):
        logging.info("Updating extension modules for macOS...")
        self.ext_modules.append(
            Extension(
                'fealpy.solver.pangulu._pangulu_r64_cpu',
                sources=['fealpy/solver/pangulu/_pangulu_r64.pyx'],
                libraries=['pangulu_r64_cpu', 'metis', 'openblas'],
                library_dirs=[os.path.expanduser('~/.local/lib')],
                include_dirs=[os.path.expanduser('~/.local/include')],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']
            ),
        )

    def install_libraries(self, datatype, platform):
        logging.info(f"Installing libraries for {datatype} on {platform} (macOS)...")
        super().install_libraries(datatype, platform)

    def install_headers(self, datatype):
        logging.info(f"Installing headers for {datatype} (macOS)...")
        super().install_headers(datatype)

    def build(self):
        logging.info("Building Pangulu libraries for macOS...")
        super().build()



class Installer:
    """
    Factory class to generate the appropriate platform-specific installer based on the current operating system.
    """

    def __new__(cls, dep, ext_modules):
        """
        Creates the appropriate installer instance based on the current operating system.
        
        Parameters:
            dep (dict): Dependency information.
            ext_modules (list): List of extension modules.

        Returns:
            BaseInstaller: The platform-specific installer.
        """
        current_platform = plat.system().lower()

        if current_platform == 'linux':
            logging.info("Detected Linux platform.")
            return LinuxInstaller(dep, ext_modules)
        elif current_platform == 'windows':
            logging.info("Detected Windows platform.")
            return WindowsInstaller(dep, ext_modules)
        elif current_platform == 'darwin':
            logging.info("Detected macOS platform.")
            return MacInstaller(dep, ext_modules)
        else:
            raise ValueError(f"Unsupported platform: {current_platform}")

def build(dep, ext_modules):
    """
    """
    if dep["compile_from_source"]:
        installer = Installer(dep, ext_modules)
        installer.build()



if __name__ == "__main__":
    # add the current directory to the path 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    dep = {
        "name": "pangulu",
        "env": "WITH_PANGULU",
        "compile_from_source": True,
        "version": "4.2.0",
        "source_url": [
            "git@github.com:SuperScientificSoftwareLaboratory/PanguLU.git",
            "https://www.ssslab.cn/assets/panguLU/PanguLU-4.2.0.zip"
        ],
        "install_prefix": "~/.local"
        }
    ext_modules = []

    # Use the factory to create the appropriate installer for the current platform
    installer = Installer(dep, ext_modules)
    installer.build()  # Run the build process
