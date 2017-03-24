'''

    @brief  Solve the benchmark "differentially heated cavity" natural convection problem using finite elements.

    @detail
        
        Solve the natural convection test problem from

            @article
            {danaila2014newton,
              title={A Newton method with adaptive finite elements for solving phase-change problems with natural convection},
              author={Danaila, Ionut and Moglan, Raluca and Hecht, Fr{\'e}d{\'e}ric and Le Masson, St{\'e}phane},
              journal={Journal of Computational Physics},
              volume={274},
              pages={826--840},
              year={2014},
              publisher={Academic Press}
            }
        
        Rather than implementing the Newton linearized system derived in danaila2014newton, allow FEniCS to automatically handle the nonlinear system.
        
        Match the notation in danaila2014newton as best as possible.

    
    @author Alexander G. Zimmerman <zimmerman@aices.rwth-aachen.de>
    
'''

from fenics import \
    UnitSquareMesh, FiniteElement, triangle, MixedElement, \
    FunctionSpace, Function, TestFunctions, split, \
    DirichletBC, Constant, \
    dot, grad, sym, div, \
    File, \
    Progress, set_log_level, PROGRESS, \
    solve
    

# Set parameters
Ra = 1.e6

Pr = 0.71

Re = 1.

theta_h = 0.5

theta_c = -0.5

K = 1

final_time = 1.0

num_time_steps = 2

gamma = 1.e-14

g = (0., -1., 0.)

global_mesh_bisection_levels = 2


# Compute derived parameters
dt = final_time / num_time_steps


# Create mesh
nc = 2**global_mesh_bisection_levels
mesh = UnitSquareMesh(nc, nc)


# Define function space for the system
P2 = FiniteElement('P', triangle, 2) # Velocity

P1 = FiniteElement('P', triangle, 1) # Pressure or Temperature; @todo: Why does danaila2014newton use space Q for pressure?

element = MixedElement([P2, P1, P1])

V = FunctionSpace(mesh, element)


# Define function and test functions
w = Function(V)

w_n = Function(V)

v, q, phi = TestFunctions(V)

u, p, theta = split(w)


# Define boundaries
hot_wall = 'near(x[0],  0.)'

cold_wall = 'near(x[0],  1.)'

adiabatic_wall = 'near(x[1],  0.) | near(x[1],  1.)'

# Define boundary conditions
element = MixedElement([P2, P1])

bc = [ \
    DirichletBC(V, Constant((0., 0., 0., theta_h)), hot_wall), \
    DirichletBC(V, Constant((0., 0., 0., theta_c)), cold_wall), \
    DirichletBC(V.sub(0), Constant((0., 0.)), adiabatic_wall), \
    DirichletBC(V.sub(1), Constant((0.)), adiabatic_wall)]
    

# Define expressions needed for variational format
Ra = Constant(Ra)
Pr = Constant(Pr)
Re = Constant(Re)
K = Constant(K)

def f_b(_theta):
    _theta*Ra/(Pr*Re*Re)

# Define variational form
def D(_u):
    return sym(grad(u))
   
   
def a(_mu, _u, _v):
    return 2.*_mu*inner(D(_u), D(_v))
    

def b(_u, _q):
    return -div(_u)*_q
    

def c(_w, _z, _v):
    return dot(div(_w)*_z, _v)
    
    
F = \
    b(u, q) - gamma*p*q \
    + dot(u, v)/dt + c(u, u, v) + a(mu, u, v) + b(v, p) - dot(f_B(theta), v) - dot(u_n, v)/dt \
    + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt
    
# Create VTK file for visualization output
solution_file = File('solution.pvd')

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Solve each time step
for n in range(num_time_steps):

    t = n*dt # Update current time

    solve(F == 0, w, bc) # Solve nonlinear problem for this time step
    
    solution_file << (w, t) # Save solution to file
    
    w_n.assign(w) # Update previous solution
    
    progress.update(t / final_time)
    