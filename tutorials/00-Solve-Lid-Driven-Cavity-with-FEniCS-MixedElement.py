import fenics


N = initial_uniform_cell_count = 4

mesh = fenics.UnitSquareMesh(N, N)


P2 = fenics.VectorElement('P', mesh.ufl_cell(), 2)

P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)

mixed_element = fenics.MixedElement([P2, P1])

W = fenics.FunctionSpace(mesh, mixed_element)


psi_u, psi_p = fenics.TestFunctions(W)


w = fenics.Function(W)

u, p = fenics.split(w)


dynamic_viscosity = 0.01

mu = fenics.Constant(dynamic_viscosity)


inner, dot, grad, div, sym =     fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
dx = fenics.dx


convection = dot(psi_u, dot(grad(u), u))

internal_source = -div(psi_u)*p

diffusion =  2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

incompressibility = -psi_p*div(u)
        
F = (incompressibility + convection + internal_source + diffusion)*dx


JF = fenics.derivative(F, w, fenics.TrialFunction(W))


lid_velocity = (1., 0.)

fixed_wall_velocity = (0., 0.)

lid_location = "near(x[1],  1.)"

fixed_wall_locations = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"

V = W.sub(0)

boundary_conditions = [
    fenics.DirichletBC(V, lid_velocity, lid_location),
    fenics.DirichletBC(V, fixed_wall_velocity, fixed_wall_locations)]


problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)


M = u[0]**2*dx

epsilon_M = 1.e-4


solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

solver.solve(epsilon_M)
