'''

@brief Solve the Navier-Stokes equations.

@detail

    This is copied from Solving PDEs in Python: The FEniCS Tutorial I.
    
'''

from fenics import *
from mshr import *
import numpy as np


T = 1. # final time

num_steps = 1000 # number of time steps

dt = T / num_steps # time step size

mu = 1. # dynamic viscosity

rho = 1. # density

lid_velocity = 1.

mesh_M = 20


# Create mesh
mesh = UnitSquareMesh(mesh_M, mesh_M)


# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)

Q = FunctionSpace(mesh, 'P', 1)


# Define boundaries
lid = 'near(x[1],  1.)'
    
not_lid = '! ' + lid

bottom = 'near(x[1], 0.)'


# Define boundary conditions    
bcu  = [ \
    DirichletBC(V, Constant((lid_velocity, 0.)), lid), \
    DirichletBC(V, Constant((0., 0.)), not_lid)]
    
bcp = [DirichletBC(Q, Constant(0.), bottom)]


# Define trial and test functions
u = TrialFunction(V)

v = TestFunction(V)

p = TrialFunction(Q)

q = TestFunction(Q)


# Define functions for solutions at previous and current time steps
u_n = Function(V)

u_ = Function(V)

p_n = Function(Q)

p_ = Function(Q)


# Define expressions used in variational forms
U = 0.5*(u_n + u)

n = FacetNormal(mesh)

f = Constant((0, 0))

k = Constant(dt)

mu = Constant(mu)


# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))


# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
    + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx

a1 = lhs(F1)

L1 = rhs(F1)


# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx

L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx


# Define variational problem for step 3
a3 = dot(u, v)*dx

L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx


# Assemble matrices
A1 = assemble(a1)

A2 = assemble(a2)

A3 = assemble(a3)


# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]

[bc.apply(A2) for bc in bcp]


# Create XDMF files for visualization output
output_dir = 'navier_stokes_lid_driven/'

xdmffile_u = XDMFFile(output_dir+'velocity.xdmf')

xdmffile_p = XDMFFile(output_dir+'pressure.xdmf')


# Create progress bar
progress = Progress('Time-stepping')

set_log_level(PROGRESS)


# Time-stepping
t = 0

for n in range(num_steps):
    # Update current time
    t += dt
    
    
    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    
    [bc.apply(b1) for bc in bcu]
    
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
    
    
    # Step 2: Pressure correction step
    b2 = assemble(L2)
    
    [bc.apply(b2) for bc in bcp]
    
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    
    
    # Step 3: Velocity correction step
    b3 = assemble(L3)
    
    solve(A3, u_.vector(), b3, 'cg', 'sor')
    
    
    # Plot solution
    plot(u_, title='Velocity')
    
    plot(p_, title='Pressure')
    
    
    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    
    xdmffile_p.write(p_, t)
    
    
    # Update previous solution
    u_n.assign(u_)
    
    p_n.assign(p_)
    
    
    # Update progress bar
    progress.update(t / T)
    
    print('u max:', u_.vector().array().max())
    
    
# Hold plot
interactive()