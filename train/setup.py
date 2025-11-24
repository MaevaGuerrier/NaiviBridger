from setuptools import setup, find_packages

setup(
    name="vint_train",
    version="0.1.0",
    package_data={
        '':['*.yaml'],
    },
    packages=find_packages(),
)
