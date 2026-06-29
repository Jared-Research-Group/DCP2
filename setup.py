from setuptools import setup
from Cython.Build import cythonize

setup(
    name="xiris",
    ext_modules=cythonize("xiris.pyx"),
)

setup(
    name="thermography",
    ext_modules=cythonize("thermography.pyx"),
)

setup(
    name="align_data",
    ext_modules=cythonize("align_data.pyx"),
)

setup(
    name="data_manipulation",
    ext_modules=cythonize("core_scripts/data_manipulation.pyx"),
)

setup(
    name="lembox",
    ext_modules=cythonize("core_scripts/lembox.pyx"),
)