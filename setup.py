from setuptools import setup, find_packages


def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]
 

setup(
    name="OptiNet",
    version="0.1.3",
    packages=['optima'], 
    install_requires= parse_requirements('requirements.txt'),
    author="Vishwanath Akuthota ,Ganesh thota and Krishna Avula",
    description='Optima is a Python library for optimizing traditional machine learning models.',
    
)
