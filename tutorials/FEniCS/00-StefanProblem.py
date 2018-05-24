import fenics


N = 1000

mesh = fenics.UnitIntervalMesh(N)


P1 = fenics.FiniteElement('P', mesh.ufl_cell(), 1)

V = fenics.FunctionSpace(mesh, P1)


psi = fenics.TestFunction(V)

T = fenics.Function(V)


stefan_number = 0.045

Ste = fenics.Constant(stefan_number)


regularization_central_temperature = 0.

T_r = fenics.Constant(regularization_central_temperature)

regularization_smoothing_parameter = 0.005

r = fenics.Constant(regularization_smoothing_parameter)

tanh = fenics.tanh

def phi(T):
    
    return 0.5*(1. + fenics.tanh((T_r - T)/r))


hot_wall_temperature = 1.

T_h = fenics.Constant(hot_wall_temperature)

cold_wall_temperature = -0.01

T_c = fenics.Constant(cold_wall_temperature)


initial_melt_thickness = 10./float(N)

T_n = fenics.interpolate(
    fenics.Expression(
        "(T_h - T_c)*(x[0] < x_m0) + T_c",
        T_h = hot_wall_temperature, 
        T_c = cold_wall_temperature,
        x_m0 = initial_melt_thickness,
        element = P1),
    V)

    
timestep_size = 1.e-2

Delta_t = fenics.Constant(timestep_size)

T_t = (T - T_n)/Delta_t

phi_t = (phi(T) - phi(T_n))/Delta_t


dot, grad = fenics.dot, fenics.grad
    
F = (psi*(T_t - 1./Ste*phi_t) + dot(grad(psi), grad(T)))*fenics.dx


JF = fenics.derivative(F, T, fenics.TrialFunction(V))


hot_wall = "near(x[0],  0.)"

cold_wall = "near(x[0],  1.)"

boundary_conditions = [
    fenics.DirichletBC(V, hot_wall_temperature, hot_wall),
    fenics.DirichletBC(V, cold_wall_temperature, cold_wall)]


problem = fenics.NonlinearVariationalProblem(F, T, boundary_conditions, JF)


solver = fenics.NonlinearVariationalSolver(problem)


solver.solve()

for timestep in range(10):
    
    T_n.vector()[:] = T.vector()
    
    solver.solve()
    