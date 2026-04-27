from setuptools import setup, find_packages

setup(
    name='mathgap',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'click==8.1.7',
        'pandas==2.2.2',
        'pydantic==2.9.2',
        'matplotlib==3.9.2',
        'networkx==3.3',
        'pyvis==0.3.2',
    ],
    python_requires='>=3.7'
)