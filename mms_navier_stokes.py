import sympy as sp
from sympy import physics as spp
from sympy.physics import vector as sppv


# Set manufactured solution
r0, beta = sp.symbols('r0, beta')

t = sp.symbols('t')

R = sppv.ReferenceFrame('R')

x = R[0]

y = R[1]

u = [0, 0]

u[0] = -(sp.exp(-beta*t**2) - 1)*(sp.sqrt(x**2 + y**2) - r0)*sp.sin(sp.atan2(y, x))

u[1] = (sp.exp(-beta*t**2) - 1)*(sp.sqrt(x**2 + y**2) - r0)*sp.cos(sp.atan2(y, x))

p = -(sp.exp(-beta*t**2) - 1)*(sp.sqrt(x**2 + y**2) - r0)**2


# Print in muparser format
print("\nManufactured solution:")

print(("Function expression = "+str(u[0])+"; "+str(u[1])+"; "+str(p)).replace('**', '^').replace('R_', ''))


# Derive manufactured source term
# @todo For now I'm not taking the symmetric part of the stress tensor, because this is complicating the symbolic implementation.
Re, mu = sp.symbols('Re, mu')

gamma = sp.symbols('gamma')

grad_p = sppv.gradient(p, R).to_matrix(R)

f0 = sp.diff(u[0], t) + sppv.dot(u[0]*R.x + u[1]*R.y, sppv.gradient(u[0], R)) - sppv.divergence(mu*sppv.gradient(u[0], R), R) + grad_p[0]

f1 = sp.diff(u[1], t) + sppv.dot(u[0]*R.x + u[1]*R.y, sppv.gradient(u[1], R)) - sppv.divergence(mu*sppv.gradient(u[1], R), R) + grad_p[1]

f2 = sppv.divergence(u[0]*R.x + u[1]*R.y, R) + gamma*p


# Print in muparser format
print("\nDerived manufactured source:")

print(("Function expression = "+str(f0)+"; "+str(f1)+"; "+str(f2)).replace('**', '^').replace('R_', ''))