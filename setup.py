from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This repository contains code for the master thesis.',
    author='Adrian Stańdo',
    author_email="adrsnek11@live.com",
    license='MIT',
    install_requires=required
)
