
from setuptools import setup, find_packages

setup(
    name='sensors',
    version='1.0',
    description='A package for sensor classes.',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'numpy',
        'opencv-python',
    ],
)
