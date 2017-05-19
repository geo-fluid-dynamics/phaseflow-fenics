# -*- coding: utf-8 -*-

import setuptools


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setuptools.setup(
    name='phaseflow',
    version='0.1.0',
    description='Simulate convective and conducive heat transfer in a phase-change material domain',
    long_description=readme,
    author='Alexander G. Zimmerman',
    author_email='alex.g.zimmerman@gmail.com',
    url='https://github.com/alexanderzimmerman/phaseflow-fenics',
    license=license,
    packages=setuptools.find_packages(exclude=('tests', 'docs'))
)