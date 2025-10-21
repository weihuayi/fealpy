import os
import json
import urllib.request
import zipfile
import subprocess
from pathlib import Path
import shutil
import platform

EXTERNAL_DIR = Path(__file__).parent
CEC_DIR = EXTERNAL_DIR.parent / "fealpy" / "opt" / "model" / "single"
os.makedirs(CEC_DIR, exist_ok=True)

def download_cec_data_zip(url: str) -> None:
    """
    Download and extract the CEC dataset zip file.

    Downloads the CEC dataset from the specified URL, extracts it to the target directory,
    and organizes the extracted folders. Skips download if data already exists.

    Parameters:
        url (str): The URL from which to download the CEC dataset zip file.

    Returns:
        None: This function does not return any value.

    Notes:
        - Checks for existing data in expected folders before downloading
        - Automatically cleans up temporary files and folders after extraction
        - Expected folders: 'input_data17', 'input_data20', 'input_data22'
    """
    zip_path = CEC_DIR / "cec_input_data.zip"

    # Check if the data already exists
    expected_folders = ["input_data17", "input_data20", "input_data22"]
    if all((CEC_DIR / f).exists() for f in expected_folders):
        print(f"[INFO] CEC data already exists in {CEC_DIR}")
        return

    print(f"[INFO] Downloading CEC data zip from {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(CEC_DIR)

    root_folder = CEC_DIR / "fealpy-cec-input_data-main"
    for folder in expected_folders:
        src = root_folder / folder
        dst = CEC_DIR / folder
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    os.remove(zip_path)
    if root_folder.exists():
        shutil.rmtree(root_folder)

    print(f"[INFO] CEC data downloaded into: {CEC_DIR}")

class CECSuite:
    """
    A class representing a CEC (Computational Electromagnetics Code) suite for building dynamic libraries.

    This class handles the compilation process for C++ files into platform-specific dynamic libraries
    (.dll on Windows, .dylib on macOS, .so on Linux) with appropriate compiler flags and settings.

    Parameters:
        name (str): The name identifier for this CEC suite.
        env (str): Environment variable name that controls whether to build this suite.
        cpp_files (list[str]): List of C++ source file names to be compiled.
        output_base (str): Base name for the output dynamic library file.

    Attributes:
        name (str): The name identifier for this CEC suite.
        env (str): Environment variable name for build control.
        cpp_files (list[str]): List of C++ source files to compile.
        output_base (str): Base name for the output library file.
    """
    
    def __init__(self, name: str, env: str, cpp_files: list[str], output_base: str):
        self.name = name
        self.env = env
        self.cpp_files = cpp_files
        self.output_base = output_base

    def build(self) -> bool:
        """
        Build the dynamic library from C++ source files.

        Compiles the specified C++ files into a platform-specific dynamic library using g++
        with optimization flags and appropriate compilation settings.

        Returns:
            bool: True if compilation was successful and output file exists, False otherwise.

        Raises:
            subprocess.CalledProcessError: If the compilation process fails.
        """
        print(f"\n[{self.name}] Start building ...")

        cpp_paths = [str(CEC_DIR / f) for f in self.cpp_files]
        system = platform.system()
        if system == "Windows":
            output_file = CEC_DIR / f"{self.output_base}.dll"
            compile_cmd = ["g++", "-O3", "-shared", "-Wall",
                        "-Wno-unused-variable", "-Wno-maybe-uninitialized", "-Wno-unused-result",
                        "-std=c++17", *cpp_paths, "-o", str(output_file)]
        elif system == "Darwin":
            output_file = CEC_DIR / f"{self.output_base}.dylib"
            compile_cmd = ["g++", "-O3", "-shared", "-Wall", "-fPIC",
                        "-Wno-unused-variable", "-Wno-maybe-uninitialized", "-Wno-unused-result",
                        "-std=c++17", *cpp_paths, "-o", str(output_file)]
        else:  # Linux
            output_file = CEC_DIR / f"{self.output_base}.so"
            compile_cmd = ["g++", "-O3", "-shared", "-Wall", "-fPIC",
                        "-Wno-unused-variable", "-Wno-maybe-uninitialized", "-Wno-unused-result",
                        "-std=c++17", *cpp_paths, "-o", str(output_file)]

        print(f"[{self.name}] Compile Command: {' '.join(compile_cmd)}")
        try:
            subprocess.run(compile_cmd, check=True)
            print(f"[{self.name}] Compilation successful, generated file: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[{self.name}] Compilation failed !")
            print(e)
            return False

        if output_file.exists():
            print(f"[{self.name}] Verified: dynamic library available.\n")
            return True
        else:
            print(f"[{self.name}] Verification failed: file missing.\n")
            return False

def build(dep, ext_modules) -> None:
    """
    Build CEC suites based on configuration and environment variables.

    Reads the build configuration from config.json, downloads CEC data if enabled,
    and compiles CEC suites according to environment variable settings.

    Parameters:
        dep: Dependency information (unused in current implementation).
        ext_modules: External modules information (unused in current implementation).

    Returns:
        None: This function does not return any value.

    Notes:
        - WITH_CEC environment variable controls overall CEC functionality
        - Individual suite environment variables control specific suite compilation
        - Configuration is read from config.json in the EXTERNAL_DIR
    """
    config_file = EXTERNAL_DIR / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Check if WITH_CEC is turned on
    with_cec = os.getenv("WITH_CEC")
    
    # Download CEC data (as long as WITH_CEC=1, download)
    if with_cec:
        for item in config:
            if "data_url" in item:
                download_cec_data_zip(item["data_url"])
                break

    # Compile CEC suite
    suites = []
    for item in config:
        if "cec_suites" in item:
            for s in item["cec_suites"]:
                suites.append(CECSuite(s["name"], s.get("env", ""), s["cpp_files"], s["output_base"]))

    for suite in suites:
        if with_cec:
            suite.build()
        elif suite.env and os.getenv(suite.env):
            suite.build()