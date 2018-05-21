import fenics
import matplotlib


N = 4

mesh = fenics.UnitSquareMesh(N, N)


P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)

P2 = fenics.VectorElement('P', mesh.ufl_cell(), 2)

mixed_element = fenics.MixedElement([P1, P2, P1])

W = fenics.FunctionSpace(mesh, mixed_element)


psi_p, psi_u, psi_T = fenics.TestFunctions(W)

w = fenics.Function(W)

p, u, T = fenics.split(w)


dynamic_viscosity = 1.

mu = fenics.Constant(dynamic_viscosity)

prandtl_number = 0.71

Pr = fenics.Constant(prandtl_number)

rayleigh_number = 1.e6

Ra = fenics.Constant(rayleigh_number)


f_B = Ra/Pr*T*fenics.Constant((0., -1.))


hot_wall_temperature = 0.5

T_h = fenics.Constant(hot_wall_temperature)

cold_wall_temperature = -0.5

T_c = fenics.Constant(cold_wall_temperature)


w_n = fenics.interpolate(
    fenics.Expression(("0.", "0.", "0.", "T_h + x[0]*(T_c - T_h)"), 
                      T_h = hot_wall_temperature, T_c = cold_wall_temperature,
                      element = mixed_element),
    W)

p_n, u_n, T_n = fenics.split(w_n)


timestep_size = 0.001

Delta_t = fenics.Constant(timestep_size)

u_t = (u - u_n)/Delta_t

T_t = (T - T_n)/Delta_t


inner, dot, grad, div, sym = \
    fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
mass = -psi_p*div(u)

momentum = dot(psi_u, u_t + dot(grad(u), u) + f_B) - div(psi_u)*p \
    + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))

energy = psi_T*T_t + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
F = (mass + momentum + energy)*fenics.dx


penalty_stabilization_parameter = 1.e-7

gamma = fenics.Constant(penalty_stabilization_parameter)

F += -psi_p*gamma*p*fenics.dx


JF = fenics.derivative(F, w, fenics.TrialFunction(W))


hot_wall = "near(x[0],  0.)"

cold_wall = "near(x[0],  1.)"

adiabatic_walls = "near(x[1],  0.) | near(x[1],  1.)"

walls = hot_wall + " | " + cold_wall + " | " + adiabatic_walls


W_u = W.sub(1)

W_T = W.sub(2)

boundary_conditions = [
    fenics.DirichletBC(W_u, (0., 0.), walls),
    fenics.DirichletBC(W_T, hot_wall_temperature, hot_wall),
    fenics.DirichletBC(W_T, cold_wall_temperature, cold_wall)]
    
    
problem = fenics.NonlinearVariationalProblem(F, w, boundary_conditions, JF)


M = inner(grad(T), grad(T))*fenics.dx

epsilon_M = 0.05


solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

w.leaf_node().vector()[:] = w_n.leaf_node().vector()

solver.solve(epsilon_M)


for timestep in range(4):
    
    w_n.leaf_node().vector()[:] = w.leaf_node().vector()
    
    timestep_size *= 2.
    
    Delta_t.assign(timestep_size)
    
    solver.solve(epsilon_M)
