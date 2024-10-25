from setuptools import setup, find_packages


def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]
 

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name="OptiNet",
    version="0.1.4",
    packages=['optima'], 
    install_requires= [],
    author="Vishwanath Akuthota ,Ganesh thota and Krishna Avula",
    description='Optima is a Python library for optimizing traditional machine learning models.',
    long_description=long_description,
    long_description_content_type="text/markdown",  # Specify Markdown format

)


