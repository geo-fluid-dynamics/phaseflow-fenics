import sympy as sp
from sympy import physics as spp
from sympy.physics import vector as sppv

from fenics import Point, RectangleMesh

import nsb_pcm as ns


dim = 2


# Set manufactured solution
u0, v0, p0, epsilon = sp.symbols('u0, v0, p0, epsilon')

t = sp.symbols('t')

R = sppv.ReferenceFrame('R')

x = R[0]

y = R[1]

u = [0, 0]

u[0] = u0*(sp.sin(x**2 + y**2) + epsilon)

u[1] = v0*(sp.cos(x**2 + y**2) + epsilon)

p = p0*(sp.sin(x**2 + y**2) + 2)


# Derive manufactured source term
# @todo For now I'm not taking the symmetric part of the stress tensor, because this is complicating the symbolic implementation.
Re, mu = sp.symbols('Re, mu')

gamma = sp.symbols('gamma')

grad_p = sppv.gradient(p, R).to_matrix(R)

f0 = sp.diff(u[0], t) + sppv.dot(u[0]*R.x + u[1]*R.y, sppv.gradient(u[0], R)) - sppv.divergence(mu*sppv.gradient(u[0], R), R) + grad_p[0]

f1 = sp.diff(u[0], t) + sppv.dot(u[0]*R.x + u[1]*R.y, sppv.gradient(u[1], R)) - sppv.divergence(mu*sppv.gradient(u[1], R), R) + grad_p[1]

f2 = sppv.divergence(u[0]*R.x + u[1]*R.y, R) + gamma*p


# Substitute values for parameters
symbolic_expressions = [u[0], u[1], p, f0, f1, f2]

sub_symbolic_expressions = [e.subs(u0, 1.).subs(v0, 1.).subs(p0, 1.).subs(epsilon, 0.001).subs(Re, 1.).subs(mu, 0.5).subs(gamma, 1.e-7) for e in symbolic_expressions]


# Set initial value expressions
iv_expressions = [0.01*e for e in sub_symbolic_expressions[0:3]]


# Convert to strings that can later be converted to UFL expressions for fenics
ufl_strings = [str(e).replace('R_x', 'x[0]').replace('R_y', 'x[1]') for e in sub_symbolic_expressions+iv_expressions]

for d in range(dim):
    ufl_strings = [e.replace('x['+str(d)+']**2', 'pow(x['+str(d)+'], 2)') for e in ufl_strings]
    
u0, u1, p, f0, f1, f2, u00, u10, p0 = ufl_strings

    
# Run the FE solver
on_wall = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)'
     
for m in (4, 8, 16):
    
    ns.run(linearize = False, \
        adaptive_time = False, \
        final_time = 100., \
        time_step_size = 10., \
        output_dir="output/mms_navier_stokes_m"+str(m), \
        mesh=RectangleMesh(Point(-0.1, 0.2), Point(0.7, 0.8), m, m, "crossed"), \
        mu = 0.5,\
        s_u = (f0, f1), \
        s_p = f1, \
        s_theta ='0.', \
        initial_values_expression = (u00, u10, p0, '0.'), \
        bc_expressions = [[0, (u0, u1), 3, on_wall], [1, p, 2, on_wall]], \
        exact_solution_expression = (u0, u1, p, '0.')) 
