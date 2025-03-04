import argparse
import subprocess
import sys
import platform
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install FEALPy")
    parser.add_argument("--with-mumps", default=0, 
                        help="Install with MUMPS")
    parser.add_argument("--with-pandulu", default=0, 
                        help="Install with Pandulu")
    parder.add_argument("--with-p4est", default=0, 
                        help="Install with p4est")
    args = parser.parse_args()
