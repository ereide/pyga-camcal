# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyga-camcal',
    version='0.1.2',
    description='Camera Calibration package',
    long_description=readme,
    author='Eivind Roson Eide',
    author_email='contact@eivindeide.me',
    url='https://github.com/ereide/GeometricAlgebra',
    license=license,
    packages=['pygacal','pygacal.camera','pygacal.common','pygacal.geometry','pygacal.rotation']
)
