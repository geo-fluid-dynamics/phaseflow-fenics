# -*- coding: utf-8 -*-
import setuptools


with open("README.md") as f:

    readme = f.read()

with open("LICENSE.txt") as f:

    license = f.read()

setuptools.setup(
    name = "phaseflow",
    version = "0.8.2-alpha",
    description = "Simulate convection-coupled melting and solidification of phase-change materials",
    long_description = readme,
    author= "Alexander G. Zimmerman",
    author_email = "zimmerman@aices.rwth-aachen.de",
    url = "https://github.com/geo-fluid-dynamics/phaseflow-fenics",
    license = license,
    packages = ["phaseflow"],
    classifiers=[
        "Programming Language :: Python :: 3.5",
    ],
)
