from setuptools import setup, find_packages

setup(
    name='modelutils',
    version='0.1.0',
    url='https://github.com/mpearmain/ModelUtils',
    packages=find_packages(),
    description='Kaggle helper functions package',
    install_requires=[
        "numpy >= 1.11.1",
        "pandas >= 0.19.0",
        "scipy >= 0.18.1",
        "scikit-learn >= 0.18.0",
    ],
)