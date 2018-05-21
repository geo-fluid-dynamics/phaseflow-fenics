import fenics
import matplotlib


N = 4

mesh = fenics.UnitSquareMesh(N, N)


P2 = fenics.VectorElement('P', mesh.ufl_cell(), 2)

P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)

P2P1 = fenics.MixedElement([P2, P1])

W = fenics.FunctionSpace(mesh, P2P1)


psi_u, psi_p = fenics.TestFunctions(W)

w = fenics.Function(W)

u, p = fenics.split(w)


dynamic_viscosity = 0.01

mu = fenics.Constant(dynamic_viscosity)


inner, dot, grad, div, sym =     fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

momentum = dot(psi_u, dot(grad(u), u)) - div(psi_u)*p     + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

mass = -psi_p*div(u)
        
F = (momentum + mass)*fenics.dx


JF = fenics.derivative(F, w, fenics.TrialFunction(W))


lid_velocity = (1., 0.)

lid_location = "near(x[1],  1.)"

fixed_wall_velocity = (0., 0.)

fixed_wall_locations = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"


V = W.sub(0)

boundary_conditions = [
    fenics.DirichletBC(V, lid_velocity, lid_location),
    fenics.DirichletBC(V, fixed_wall_velocity, fixed_wall_locations)]

problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)


M = u[0]**2*fenics.dx

epsilon_M = 1.e-4


solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

solver.solve(epsilon_M)
