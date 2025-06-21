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
        self.ext_modules.append(
            Extension(
                'fealpy.solver.pangulu._pangulu_r32_cpu',
                sources=['fealpy/solver/pangulu/_pangulu_r32.pyx'],
                libraries=['pangulu_r32_cpu', 'metis', 'openblas'],
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

        logging.info(f"Installing libraries for {datatype} on {platform}...")

        install_prefix = os.path.expanduser(self.dep['install_prefix'])
        fname = f"lib/libpangulu_{datatype.lower()}_{platform.lower()}"

        # move dynamic library to install_prefix
        source = "lib/libpangulu.so"
        if not os.path.exists(source):
            raise FileNotFoundError(f"{source} doesn't exist!")

        target = os.path.join(install_prefix, fname + '.so')
        shutil.move(source, target)

        # move static library to install_prefix
        source = "lib/libpangulu.a"
        target = os.path.join(install_prefix, fname + '.a')
        if not os.path.exists(source):
            raise FileNotFoundError(f"{source} doesn't exist!")
        shutil.move(source, target)

    def install_headers(self, datatype):
        """
        Install the appropriate headers for the specified datatype (R64 or R32).

        Parameters:
            datatype (str): The data type, should be 'R64' or 'R32'.
        """
        logging.info(f"Installing headers for {datatype}...")

        install_prefix = os.path.expanduser(self.dep['install_prefix'])
        iheader = """
#ifndef PANGULU_H
#define PANGULU_H

#include <stdio.h>
#include <stdlib.h>

typedef struct pangulu_init_options
{
    int nthread;
    int nb;
}pangulu_init_options;

typedef struct pangulu_gstrf_options
{
}pangulu_gstrf_options;

typedef struct pangulu_gstrs_options
{
}pangulu_gstrs_options;

typedef unsigned long long int sparse_pointer_t;
typedef unsigned int sparse_index_t;
    """
        icontent = """ 
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    void pangulu_init(sparse_index_t pangulu_n, sparse_pointer_t pangulu_nnz, sparse_pointer_t *csr_rowptr, sparse_index_t *csr_colidx, sparse_value_t *csr_value, pangulu_init_options *init_options, void **pangulu_handle);
    void pangulu_gstrf(pangulu_gstrf_options *gstrf_options, void **pangulu_handle);
    void pangulu_gstrs(sparse_value_t *rhs, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
    void pangulu_gssv(sparse_value_t *rhs, pangulu_gstrf_options *gstrf_options, pangulu_gstrs_options *gstrs_options, void **pangulu_handle);
    void pangulu_finalize(void **pangulu_handle);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // PANGULU_H
    """
        if datatype == 'R64':
            value_type = """
typedef double sparse_value_t;
            """
        elif datatype == 'R32':
            value_type = """
typedef float sparse_value_t;
            """
        else:
            raise ValueError("datatype must be 'R64' or 'R32'!")

        fname = f"include/pangulu_{datatype.lower()}.h"
        target = os.path.join(install_prefix, fname)
        with open(target, 'w') as f:
            content = iheader + value_type + icontent
            f.write(content.strip())  # Remove leading/trailing whitespace

    def build(self):
        """
        Build and install the Pangulu libraries, headers, and generate Makefile for compilation on Linux platform.
        """
        from source_downloader import fetch 
        logging.info("Building Pangulu libraries for Linux...")

        result_path = fetch(self.dep["source_url"])

        # Update extension modules
        self.update()

        currentPath = os.getcwd()
        os.chdir(result_path)

        # Define compile flags
        generalFlags = """
COMPILE_LEVEL = -O3
CC = gcc $(COMPILE_LEVEL) #-fsanitize=address
MPICC = mpicc $(COMPILE_LEVEL) #-fsanitize=address
OPENBLAS_INC = -I/usr/include/x86_64-linux-gnu/openblas-openmp/
OPENBLAS_LIB = -L/usr/lib/x86_64-linux-gnu/openblas-openmp/ -lopenblas
MPICCFLAGS = $(OPENBLAS_INC) $(CUDA_INC) $(OPENBLAS_LIB) -fopenmp -lpthread -lm
MPICCLINK = $(OPENBLAS_LIB)
METISFLAGS =  -I/usr/include
        """

        cudaFlags = """
#0201000,GPU_CUDA
CUDA_PATH = /usr/local/cuda
CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcudart -lcusparse
NVCC = nvcc $(COMPILE_LEVEL)
NVCCFLAGS = $(PANGULU_FLAGS) -w -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 $(CUDA_INC) $(CUDA_LIB)
        """

        datatype = ['R64', 'R32']
        platform = ['cpu', 'gpu']

        mf = """
.PHONY : lib src clean update

lib : src
	$(MAKE) -C $@

src:
	$(MAKE) -C $@

clean:
	(cd src; $(MAKE) clean)
	(cd lib; $(MAKE) clean)
	(cd examples; $(MAKE) clean)

update : clean all
        """
        # Write Makefile
        with open('Makefile', 'w') as f:
            f.write(mf.strip())  # Remove leading/trailing whitespace

        for dt in datatype:
            self.install_headers(dt)
            for pf in platform:
                flag = '' if pf == 'cpu' else '-DGPU_OPEN'
                panguluFlags = f"""
PANGULU_FLAGS = -DPANGULU_LOG_INFO -DCALCULATE_TYPE_{dt} -DMETIS -DPANGULU_MC64 {flag} -DHT_IS_OPEN
                """
                # Combine compile flags
                compile_flags = generalFlags + panguluFlags + cudaFlags

                # Write make.inc file
                with open('make.inc', 'w') as f:
                    f.write(compile_flags.strip())  # Remove leading/trailing whitespace

                # Execute compilation
                os.system("make clean")  # Clean old compilation results
                os.system("make -j 4")
                self.install_libraries(dt, pf)

        os.chdir(currentPath)


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

