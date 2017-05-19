# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.txt') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='phaseflow',
    version='0.1.x',
    description='Simulate convective and conducive heat transfer in a phase-change material domain',
    long_description=readme,
    author='Alexander G. Zimmerman',
    author_email='alex.g.zimmerman@gmail.com',
    url='https://github.com/alexanderzimmerman/phaseflow-fenics',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)