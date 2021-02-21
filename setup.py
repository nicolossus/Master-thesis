import numpy
import setuptools
from setuptools.extension import Extension

setuptools.setup(
    name="pylfi",
    version="0.0.1",
    author="Nicolai Haug",
    author_email="nicolai.haug@fys.uio.no",
    description="Likelihood-free inference with Python",
    # this installs the Python package (looks for dirs containing '__init__.py')
    packages=setuptools.find_packages(),
    # 'scripts' are put on $PATH
    # scripts=["bin/instapy"],
    # Cython extensions are compiled
    # ext_modules=[Extension(
    # the 'import' name of the module
    # "instapy._cython",
    # the location of Cython source files
    # sources=["instapy/_cython.pyx"],
    # any additional included directories, e.g. for Cython
    # included_dirs=[numpy.get_include()],
    # )],
    setup_requires=["cython", "numpy", "setuptools>=18.0"],
    install_requires=["numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
