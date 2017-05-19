''' Steady Navier-Stokes lid-driven cavity example

Following instructions at http://www.nbi.dk/~mathies/cm/fenics_instructions.pdf

'''

from fenics import *

# Create mesh
mesh_M = 32

mesh = UnitSquareMesh(mesh_M, mesh_M)

# Define function spaces for the system
pressure_order = 1

velocity_order = pressure_order + 1

V = VectorFunctionSpace(mesh, 'CG', velocity_order)

P = FunctionSpace(mesh, 'CG', pressure_order) # @todo mixing up test function space

V_ele = VectorElement('CG', mesh.ufl_cell(), velocity_order)

P_ele = FiniteElement('CG', mesh.ufl_cell(), pressure_order)

W = FunctionSpace(mesh, V_ele*P_ele)

# Make trial and test functions
u, p = TrialFunctions(W)

v, q = TestFunctions(W)


# Set boundaries and boundary conditions
lid = 'near(x[1],  1.)'
    
not_lid = '! ' + lid

bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'
    
bc = [ \
    DirichletBC(W.sub(0), Constant((1., 0.)), lid), \
    DirichletBC(W.sub(0), Constant((0., 0.)), not_lid), \
    DirichletBC(W.sub(1), Constant(0.), bottom_left_corner, method='pointwise')]

        
# Set parameters
Reynolds = 1.0 # or any other value

nu = 1./Reynolds


# Create solution vectors
w = Function(W)

u, p = (as_vector((w[0], w[1])), w[2])


# Set the non-linear functional
F = inner(grad(u)*u, v)*dx \
    + nu*inner(grad(u), grad(v))*dx \
    - p*div(v)*dx \
    - q*div(u)*dx
    

# Solve 
J = derivative(F, w)

solve(F == 0, w, bc, J=J)


# Output
output_dir = 'steady_cavity/'

velocity_file = File(output_dir + 'velocity.pvd')

_velocity, _pressure = _w.split()
        
velocity_file << _velocity 