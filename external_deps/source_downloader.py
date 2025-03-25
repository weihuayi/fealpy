import os
import subprocess
import logging
import shutil
import zipfile
from pathlib import Path
from abc import ABC, abstractmethod
import platform as plat

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseSourceDownloader(ABC):
    """
    Abstract base class for platform-specific source downloaders.
    Each platform must implement the `download` and `extract` methods.
    """

    def __init__(self, source_urls, download_dir='/tmp'):
        """
        Initializes the SourceDownloader with dependencies and download directory.
        
        Parameters:
            source_urls (list): List containing the source URLs.
            download_dir (str): Directory to store downloaded files (default is '/tmp').
        """
        self.source_urls = source_urls 
        self.download_dir = Path(download_dir)

    @abstractmethod
    def download(self):
        """
        Downloads the source files from the provided URLs.
        """
        pass

    @abstractmethod
    def extract(self, file_path):
        """
        Extracts the downloaded source file if it's a compressed archive.
        """
        pass

    def is_compressed_file(self, url):
        """
        Determines whether the given URL points to a compressed file.

        Parameters:
            url (str): The URL to check.

        Returns:
            bool: True if the URL points to a compressed file, False otherwise.
        """
        # Define a list of common compressed file extensions
        compressed_extensions = [
            '.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', 
            '.gz', '.bz2', '.7z', '.xz', '.rar', '.lzma', '.Z', '.tar.lzma'
        ]
        
        # Use regex to check if the URL ends with one of the common compressed file extensions
        return any(url.endswith(ext) for ext in compressed_extensions)


class LinuxSourceDownloader(BaseSourceDownloader):
    def fetch(self):
        """
        Fetchs the source files from the provided URLs for Linux platform.
        
        Tries each URL in order until a successful download is completed.
        
        Returns:
            Path: Path to the downloaded and extracted directory.
        
        Raises:
            Exception: If all download attempts fail.
        """
        source_urls = self.source_urls
        if not source_urls:
            raise ValueError("No source URLs provided in the source_urls.")
        source_dir = []
        for url in source_urls:
            try:

                logging.info(f"Attempting to download from {url}...")

                if url.endswith('.git'):
                    source_dir = self.git_clone(url)
                    break
                elif self.is_compressed_file(url): 
                    file_path = self.download(url)
                    source_dir = self.extract(file_path)
                    break
                else:
                    raise ValueError(f"Unsupported source type for {url}")
                
            except Exception as e:
                logging.error(f"Failed to download or extract from {url}: {e}")
                continue
        if not source_dir: 
            raise Exception("All download attempts failed.")
        else:
            return source_dir

    def download(self, url):
        """
        Downloads a compressed file from the provided URL.
        
        Parameters:
            url (str): URL of the compressed file to download.
        
        Returns:
            Path: Path to the downloaded compressed file.
        
        Raises:
            Exception: If download fails.
        """
        file_name = url.split('/')[-1]
        file_path = self.download_dir / file_name
        logging.info(f"Downloading {url} to {file_path} using wget...")
        result = subprocess.run(['wget', url, '-O', str(file_path)], capture_output=True, text=True)

        if result.returncode == 0:
            logging.info(f"Successfully downloaded {file_name}.")
            return file_path
        else:
            raise Exception(f"Download failed: {result.stderr}")

    def git_clone(self, url):
        """
        Clones a git repository from the provided URL using the system's git command.
        
        Parameters:
            url (str): URL of the git repository to clone.
        
        Returns:
            Path: Path to the cloned repository directory.
        
        Raises:
            Exception: If cloning fails.
        """
        try:
            repo_name = url.split('/')[-1].replace('.git', '')
            clone_dir = self.download_dir / repo_name

            # Check if the repository already exists
            if clone_dir.exists():
                logging.info(f"Repository {repo_name} already exists at {clone_dir}. Skipping clone.")
                return clone_dir  # Return the existing directory

            logging.info(f"Cloning {url} to {clone_dir} using system git...")
            result = subprocess.run(['git', 'clone', url, str(clone_dir)], capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Successfully cloned {repo_name} into {clone_dir}.")
                return clone_dir
            else:
                raise Exception(f"Git clone failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Error cloning git repository: {e}")
            raise

    def extract(self, file_path):
        """
        Extracts the compressed archive file if it's a compressed file.
        
        Parameters:
            file_path (Path): Path to the downloaded file.
        
        Returns:
            Optional[Path]: Path to the extracted directory if successful, None otherwise.
        
        Raises:
            ValueError: If unsupported archive format is encountered.
        """
        if file_path.suffix == '.zip':
            extracted_dir = self.download_dir / file_path.stem
            logging.info(f"Extracting {file_path} to {extracted_dir}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.download_dir)
            if extracted_dir.exists():
                logging.info(f"Successfully extracted to {extracted_dir}.")
                return extracted_dir
            else:
                raise ValueError(f"Extraction {file_path} failed.")
        else:
            raise ValueError(f"Unsupported archive format: {file_path.suffix}")


class SourceDownloader:
    """
    Factory class to create platform-specific source downloader instances.
    """

    def __new__(cls, source_urls, download_dir='/tmp'):
        """
        Creates the appropriate source downloader instance based on the current operating system.
        
        Parameters:
            dep (dict): Dependency information.
            download_dir (str): Directory to store downloaded files (default is '/tmp').
        
        Returns:
            BaseSourceDownloader: Platform-specific downloader instance.
        """
        current_platform = plat.system().lower()

        if current_platform == 'linux':
            logging.info("Detected Linux platform.")
            return LinuxSourceDownloader(source_urls, download_dir)
        elif current_platform == 'windows':
            logging.info("Windows platform detected. Windows-specific download logic can be implemented.")
            # Implement Windows-specific downloader here
            raise NotImplementedError("Windows downloader not implemented yet.")
        elif current_platform == 'darwin':
            logging.info("macOS platform detected. macOS-specific download logic can be implemented.")
            # Implement macOS-specific downloader here
            raise NotImplementedError("macOS downloader not implemented yet.")
        else:
            raise ValueError(f"Unsupported platform: {current_platform}")

def fetch(source_urls):
    # Create the appropriate downloader for the current platform
    downloader = SourceDownloader(source_urls)
    try:
        source_dir = downloader.fetch()
        if source_dir:
            logging.info(f"Source successfully downloaded and extracted to: {source_dir}")
            return source_dir
        else:
            logging.error(f"Source source directory is empty!")

    except Exception as e:
        logging.error(f"Source download failed: {e}")



if __name__ == "__main__":
    source_urls = [
            "git@github.com:SuperScientificSoftwareLaboratory/PanguLU.git",
            "https://www.ssslab.cn/assets/panguLU/PanguLU-4.2.0.zip"
        ]

    # Create the appropriate downloader for the current platform
    downloader = SourceDownloader(source_urls)
    try:
        source_dir = downloader.fetch()
        if source_dir:
            logging.info(f"Source successfully downloaded and extracted to: {source_dir}")
    except Exception as e:
        logging.error(f"Source download failed: {e}")

