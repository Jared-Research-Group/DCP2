from setuptools import setup
from Cython.Build import cythonize

setup(
    name="xiris",
    ext_modules=cythonize("create_xirisvideo.pyx"),
)

setup(
    name="thermography",
    ext_modules=cythonize("thermography.pyx"),
)