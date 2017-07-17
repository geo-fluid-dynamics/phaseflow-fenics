# -*- coding: utf-8 -*-
from dolfin import VectorFunctionSpace, Function, DirichletBC
from dolfin.cpp.mesh import UnitIntervalMesh

mesh = UnitIntervalMesh(5)

V = VectorFunctionSpace(mesh, 'P', 1)
u = Function(V)

bc = DirichletBC(V, [10.0], 'x[0] < 0.5')

print(bc.get_boundary_values())