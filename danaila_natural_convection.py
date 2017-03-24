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
    UnitSquareMesh, FiniteElement, VectorElement, MixedElement, \
    FunctionSpace, VectorFunctionSpace, \
    Function, TestFunctions, split, \
    DirichletBC, Constant, \
    dot, inner, grad, sym, div, \
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

g = (0., -1.)

mu = 1

global_mesh_bisection_levels = 2

pressure_order = 1

temperature_order = 1


# Compute derived parameters
dt = final_time / num_time_steps

velocity_order = pressure_order + 1


# Create mesh
nc = 2**global_mesh_bisection_levels
mesh = UnitSquareMesh(nc, nc)


# Define function spaces for the system
VxV = VectorFunctionSpace(mesh, 'Lagrange', velocity_order)

Q = FunctionSpace(mesh, 'Lagrange', pressure_order)

V = FunctionSpace(mesh, 'Lagrange', temperature_order)

'''
MixedFunctionSpace used to be available but is now deprecated. 
The way that fenics separates function spaces and elements is confusing.
To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
'''
VxV_ele = VectorElement('Lagrange', mesh.ufl_cell(), velocity_order)
Q_ele = FiniteElement('Lagrange', mesh.ufl_cell(), pressure_order)
V_ele = FiniteElement('Lagrange', mesh.ufl_cell(), temperature_order)

W = FunctionSpace(mesh, MixedElement([VxV_ele, Q_ele, V_ele]))


# Define function and test functions
w = Function(W)

w_n = Function(W)

v, q, phi = TestFunctions(W)


# Split solution function to access variables separately
u, p, theta = split(w)

u_n, p_n, theta_n = split(w_n)


# Define boundaries
hot_wall = 'near(x[0],  0.)'

cold_wall = 'near(x[0],  1.)'

adiabatic_wall = 'near(x[1],  0.) | near(x[1],  1.)'

# Define boundary conditions

bc = [ \
    DirichletBC(W, Constant((0., 0., 0., theta_h)), hot_wall), \
    DirichletBC(W, Constant((0., 0., 0., theta_c)), cold_wall), \
    DirichletBC(W.sub(0), Constant((0., 0.)), adiabatic_wall), \
    DirichletBC(W.sub(1), Constant((0.)), adiabatic_wall)]
    

# Define expressions needed for variational format
Ra = Constant(Ra)
Pr = Constant(Pr)
Re = Constant(Re)
K = Constant(K)
mu = Constant(mu)
g = Constant(g)
dt = Constant(dt)
gamma = Constant(gamma)


# Define variational form
def f_B(_theta):
    return _theta*Ra/(Pr*Re*Re)*g
       
   
def a(_mu, _u, _v):

    def D(_u):
        return sym(grad(u))
    
    return 2.*_mu*inner(D(_u), D(_v))
    

def b(_u, _q):
    return -div(_u)*_q
    

def c(_w, _z, _v):
    return dot(div(_w)*_z, _v)
    
    
F = b(u, q) - gamma*p*q \
    + dot(u, v)/dt + c(u, u, v) + a(mu, u, v) + b(v, p) - dot(f_B(theta), v) - dot(u_n, v)/dt \
    + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt
    
print type(F)
    
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
    