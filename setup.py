import sys
import os
from setuptools import setup, find_packages

setup(
    name='bloom-bags',
    version='0.0.1',
    author='Geoff McDonald',
    author_email='glmcdona@gmail.com',
    packages=find_packages(exclude=['tests*']),
    url='http://pypi.python.org/pypi/bloom-bags/',
    license='MIT',
    description='Bloom filters for ML.',
    long_description=open('README.md').read(),
    install_requires=[
        "sklearn",
        "numpy",
        "fastbloom-rs",
        "gym==0.19.0"
    ],
    test_suite='nose2.collector.collector',
    tests_require=['nose2'],
)
