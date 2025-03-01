import platform
import subprocess
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class CudaInstallerError(Exception):
    """Custom exception for CUDA installation errors"""
    pass

class CudaInstaller(ABC):
    """Abstract base class for CUDA installers"""
    
    def __init__(self, cuda_version: str, driver_version: Optional[str] = None):
        self.cuda_version = cuda_version
        self.driver_version = driver_version
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def check_compatibility(self) -> bool:
        """Check system and hardware compatibility"""
        pass
    
    @abstractmethod
    def install_dependencies(self) -> None:
        """Install required system dependencies"""
        pass
    
    @abstractmethod
    def install_cuda(self) -> None:
        """Execute the CUDA installation process"""
        pass
    
    @abstractmethod
    def configure_environment(self) -> None:
        """Configure environment variables"""
        pass
    
    def validate_installation(self) -> bool:
        """Verify successful installation"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.cuda_version in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False

class UbuntuCudaInstaller(CudaInstaller):
    """CUDA installer implementation for Ubuntu systems"""
    
    def __init__(self, cuda_version: str, driver_version: str = "550"):
        """
        TODO:
            1. add support for Ubuntu2204
        """
        super().__init__(cuda_version, driver_version)
        self.repo_config = {
            "12.5": {
                "keyring": "cuda-keyring_1.1-1_all.deb",
                "repo_url": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
            }
        }
    
    def check_compatibility(self) -> bool:
        """Verify NVIDIA GPU presence and system version"""
        try:
            # Check for NVIDIA GPU
            lspci = subprocess.run(
                ["lspci", "-nnk"],
                capture_output=True,
                text=True,
                check=True
            )
            if "NVIDIA" not in lspci.stdout:
                raise CudaInstallerError("No NVIDIA GPU detected")
            
            # Verify Ubuntu version
            if platform.system() != "Linux" or "Ubuntu" not in platform.version():
                raise CudaInstallerError("Unsupported OS version")
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Compatibility check failed: {str(e)}")
            return False
    
    def install_dependencies(self) -> None:
        """Install system dependencies"""
        commands = [
            ["sudo", "apt", "update"],
            ["sudo", "apt", "install", "-y", "build-essential"]
        ]
        self._execute_commands(commands, "Installing dependencies")
    
    def install_cuda(self) -> None:
        """Execute CUDA installation workflow"""
        try:
            # Install drivers
            self.logger.info("Installing NVIDIA drivers...")
            subprocess.run(
                ["sudo", "apt", "install", "-y", f"nvidia-driver-{self.driver_version}"],
                check=True
            )
            
            # Configure CUDA repository
            repo_info = self.repo_config.get(self.cuda_version)
            if not repo_info:
                raise CudaInstallerError(f"Unsupported CUDA version: {self.cuda_version}")
            
            self._execute_commands([
                ["wget", f"{repo_info['repo_url']}{repo_info['keyring']}"],
                ["sudo", "dpkg", "-i", repo_info['keyring']],
                ["sudo", "apt", "update"]
            ], "Configuring CUDA repository")
            
            # Install CUDA Toolkit
            self.logger.info(f"Installing CUDA Toolkit {self.cuda_version}...")
            subprocess.run(
                ["sudo", "apt", "install", "-y", f"cuda-toolkit-{self.cuda_version.replace('.', '-')}"],
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            raise CudaInstallerError(f"Installation failed: {str(e)}")
    
    def configure_environment(self) -> None:
        """Set up environment variables"""
        env_lines = [
            f"\n# CUDA {self.cuda_version} configuration",
            f"export PATH=/usr/local/cuda-{self.cuda_version}/bin${{PATH:+:${{PATH}}}}",
            f"export LD_LIBRARY_PATH=/usr/local/cuda-{self.cuda_version}/lib64${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
        ]
        
        try:
            with open(os.path.expanduser("~/.bashrc"), "a") as f:
                f.write("\n".join(env_lines))
            self.logger.info("Environment variables added to ~/.bashrc")
        except IOError as e:
            raise CudaInstallerError(f"Failed to configure environment: {str(e)}")
    
    def _execute_commands(self, commands: List[List[str]], description: str) -> None:
        """Generic method to execute command sequences"""
        self.logger.info(description)
        for cmd in commands:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise CudaInstallerError(f"Command failed: {' '.join(cmd)}")

class CudaInstallerFactory:
    """Factory class for CUDA installers"""
    
    _installers: Dict[str, Type[CudaInstaller]] = {
        "Linux": UbuntuCudaInstaller,
        # Extension point: Add other OS implementations here
    }
    
    @classmethod
    def create_installer(cls, 
                        cuda_version: str,
                        driver_version: Optional[str] = None) -> CudaInstaller:
        """Create installer instance based on current system"""
        system = platform.system()
        installer_class = cls._installers.get(system)
        
        if not installer_class:
            raise NotImplementedError(f"No installer available for {system}")
            
        return installer_class(cuda_version, driver_version)

# Usage example
if __name__ == "__main__":
    try:
        # Create installer instance
        installer = CudaInstallerFactory.create_installer(
            cuda_version="12.5",
            driver_version="550"
        )
        
        # Execute installation workflow
        if installer.check_compatibility():
            installer.install_dependencies()
            installer.install_cuda()
            installer.configure_environment()
            
            if installer.validate_installation():
                print("CUDA installation completed successfully!")
            else:
                print("CUDA installation validation failed")
                
    except CudaInstallerError as e:
        print(f"Installation error: {str(e)}")
    except NotImplementedError as e:
        print(f"System not supported: {str(e)}")
