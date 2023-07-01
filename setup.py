"""
This file is used to build the package and upload it to PyPI
"""
# pylint: skip-file  (PyLint is not happy with importing the already built-in open)
from os import path

# To use a consistent encoding
from codecs import open

# Always prefer setuptools over distutils
from setuptools import setup


# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

# This call to setup() does all the work
setup(
    name="botiverse",
    version="0.1.0",
    description='''botiverse is a chatbot library that offers a high-level API to
    access a diverse set of chatbot models''',
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://botiverse.readthedocs.io/",
    author="Essam W., Mohamed Saad, Yousef Atef, Karim Taha",
    author_email="essamwisam@outlook.com",
    license="GPLv3",
    classifiers=[                               # https://pypi.org/classifiers/
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["botiverse", "botiverse.basic_chatbot", "botiverse.TODS"],
    include_package_data=True,
    install_requires=["numpy", "torch"]            # just as was in requirements.txt
)


# Steps to upload to PyPI
# 0 - Increment the version number in setup.py
# 1 - Remove the dist folder
# 2- python3 setup.py sdist bdist_wheel  
# 3 - twine upload dist/*

# To upload to test PyPI: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
