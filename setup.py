import sys
import os
from setuptools import setup, find_packages

setup(
    name='stratified_vectorizer',
    version='0.0.1',
    author='Geoff McDonald',
    author_email='glmcdona@gmail.com',
    packages=find_packages(exclude=['tests*']),
    url='http://pypi.python.org/pypi/stratified_vectorizer/',
    license='MIT',
    description='Bloom filters for ML.',
    long_description=open('README.md').read(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "fastbloom-rs",
        "pytest",
    ],
    test_suite='nose2.collector.collector',
    tests_require=['nose2'],
)
