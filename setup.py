from pathlib import Path
from setuptools import find_packages, setup

modules = sorted(p.stem for p in Path(__file__).parent.glob("src/*.py"))
packages = find_packages(where="src") 
setup(package_dir={"": "src"}, py_modules=modules, packages=packages)
