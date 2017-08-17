# -*- coding: utf-8 -*-

import setuptools


with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setuptools.setup(
    name='phaseflow',
    version='0.2.0',
    description='Simulate convective and conducive heat transfer in a phase-change material domain',
    long_description=readme,
    author='Alexander G. Zimmerman',
    author_email='zimmerman@aices.rwth-aachen.de',
    url='https://github.com/alexanderzimmerman/phaseflow-fenics',
    license=license,
    packages=setuptools.find_packages(exclude=('tests', 'docs'))
)