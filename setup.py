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

# requirements.txt
def read_requirements(name):
    """Read requirements from requirements.txt."""
    with open(path.join(HERE, name), encoding='utf-8') as f:
        requirements = f.read().splitlines()
    return requirements



# This call to setup() does all the work
setup(
    name="botiverse",
    version="0.5.1",
    description='''botiverse is a chatbot library that offers a high-level API to access a diverse set of chatbot models''',
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    package_dir={
        "botiverse": "botiverse",
        "botiverse.bots": "botiverse/bots",
        "botiverse.models": "botiverse/models",
        "botiverse.preprocessors": "botiverse/preprocessors",
    },
    packages=["botiverse", 
              "botiverse.bots", 
              "botiverse.bots.BasicBot", 
              "botiverse.bots.ConverseBot", 
              "botiverse.bots.VoiceBot",
              "botiverse.bots.WhizBot", 
              "botiverse.bots.BasicTaskBot",
              "botiverse.bots.TaskBot", 
              "botiverse.Theorizer",
              "botiverse.Theorizer.model",
              "botiverse.Theorizer.squad",
              "botiverse.models",
              "botiverse.preprocessors", 
              "botiverse.gui",
              "botiverse.models.BERT",
              "botiverse.models.FastSpeech1",
              "botiverse.models.GRUClassifier",
              "botiverse.models.LinearClassifier",
              "botiverse.models.LSTM",
              "botiverse.models.NN",
              "botiverse.models.SVM",
              "botiverse.models.T5Model",
              "botiverse.models.TRIPPY",
              "botiverse.preprocessors.BertEmbeddings",
              "botiverse.preprocessors.BoW",
              "botiverse.preprocessors.Frequency",
              "botiverse.preprocessors.GloVe",
              "botiverse.preprocessors.Special",
              "botiverse.preprocessors.Special.ConverseBot_Preprocessor",
              "botiverse.preprocessors.Special.WhizBot_BERT_Preprocessor",
              "botiverse.preprocessors.Special.WhizBot_GRU_Preprocessor",
              "botiverse.preprocessors.GloVe",    
              "botiverse.preprocessors.TF_IDF",
              "botiverse.preprocessors.TF_IDF_GLOVE",
              "botiverse.preprocessors.Vocalize",
              "botiverse.preprocessors.Wav2Vec",
              "botiverse.gui.static",
              "botiverse.gui.static.icons",
              "botiverse.gui.templates",
              ],
        include_package_data = True,
        package_data={
            'botiverse.Theorizer.squad':['*.txt'],
            'botiverse.models.TRIPPY': ['*.txt'],
            'botiverse.gui':['*.zip', '*.png', '*pdf', '*jpeg','*ipynb', '*html', '*css', '*pkl', '*js'],
            'botiverse.gui.static':['*.zip', '*.png', '*pdf', '*jpeg','*ipynb', '*html', '*css', '*pkl', '*js'],
            'botiverse.gui.static.icons':['*.zip', '*.png', '*pdf', '*jpeg','*ipynb', '*html', '*css', '*pkl', '*js'],
            'botiverse.gui.templates':['*.zip', '*.png', '*pdf', '*jpeg','*ipynb', '*html', '*css', '*pkl', '*js'],
            },
    install_requires=read_requirements('./requirements/requirements.txt'),
    extras_require={
        "voice": read_requirements("./requirements/requirements_voice.txt"),
    }
)


# Steps to upload to PyPI
# 0 - Increment the version number in setup.py
# 1 - Remove the dist folder
# 2- python3 setup.py sdist bdist_wheel  
# 3 - twine upload dist/*

# To upload to test PyPI: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
