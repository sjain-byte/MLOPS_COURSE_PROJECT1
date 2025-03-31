from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS_PROJECT-1",
    version="0.1",
    author="Suhani Jain",
    packages=find_packages(),
    install_requires = requirements,
)
