# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys
import numpy as np
sys.path.append('./src')
sys.path.append('./test')

ext_modules = [Extension(
        'salientdetect.detector',
        ["salientdetect/detector.pyx"],
        include_dirs= [np.get_include()],
        language="c++")
]

setup(
    name="salientdetect",
    description='Slient Region Detector from Image',
    version="1.0.0",
    long_description=open('README.rst').read(),
    packages=find_packages(),
    install_requires=["numpy", 'nose', 'cython', 'scikit-image'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    author='Shunsuke Aihara',
    author_email="aihara@argmax.jp",
    url='https://github.com/shunsukeaihara/objectdetect',
    license="MIT License",
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose', 'numpy', 'cython', 'scikit-image'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ]
)
