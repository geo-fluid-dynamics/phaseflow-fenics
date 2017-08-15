import fenics


''' @brief Example of nonlinear problem with mixed finite elements

@detail This is the well-known "natural convection of differentially heated cavity" benchmark problem,
modeled by the incompressible Navier-Stokes-Boussinesq mass and momentum equations, coupled to the
energy equation written in enthalpy form.

Here we attempt to use the scripts from section 1.2.4 of the FEniCS book.

To keep this short, we only solve the first time step, which is the hardest one anyway (because we 
haven't put much work into setting the initial values).

@todo Debug the Jacobian bilinear variational form.'''

def nonlinear_mixedfe(automatic_jacobian=True):

    # Set physical parameters
    Ra = fenics.Constant(1.e6)
    
    Pr = fenics.Constant(0.71)
    
    C = fenics.Constant(1.)
    
    K = fenics.Constant(1.)
    
    g = fenics.Constant((0., -1.))
    
    mu_sl = fenics.Constant(1.)
    
    ddtheta_mu_sl = fenics.Constant(0.)
    
    m_B = lambda theta : theta*Ra/Pr

    ddtheta_m_B = Ra/Pr
    
    
    # Set numerical parameters.
    mesh = fenics.UnitSquareMesh(10, 10, 'crossed')
    
    dt = fenics.Constant(1.e-3)
    
    gamma = fenics.Constant(1.e-7)
    
    pressure_degree = 1
    
    temperature_degree = 1
    

    # Set function spaces for the variational form     .
    velocity_degree = pressure_degree + 1

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0 '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)
    
    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element, temperature_element])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    
    # Set initial values and boundary conditions.
    hot_wall = 'near(x[0],  0.)'
    
    cold_wall = 'near(x[0],  1.)'
    
    no_slip_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.) | near(x[1], 1.)'
    
    w_n = fenics.interpolate(fenics.Expression(("0.", "0.", "0.",
        "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"), element=W_ele), W)
    
    bcs = [
        fenics.DirichletBC(W.sub(0), fenics.Expression(("0.", "0."), degree=velocity_degree + 1),
            no_slip_walls, method='topological'),
        fenics.DirichletBC(W.sub(2), fenics.Expression(("0.5"), degree=temperature_degree + 1),
            hot_wall, method='topological'),
        fenics.DirichletBC(W.sub(2), fenics.Expression(("-0.5"), degree=temperature_degree + 1),
            cold_wall, method='topological')]    
    
    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    u_n, p_n, theta_n = fenics.split(w_n)
    
    dw = fenics.TrialFunction(W)
    
    du, dp, dtheta = fenics.split(dw)
    
    v, q, phi = fenics.TestFunctions(W)
        
    w_ = fenics.Function(W)
    
    u_, p_, theta_ = fenics.split(w_)
    
    F = (
        b(u_, q) - gamma*p_*q 
        + dot(u_, v)/dt + c(u_, u_, v) + a(mu_sl, u_, v) + b(v, p_)
        + dot(m_B(theta_)*g, v)
        + C*theta_*phi/dt - dot(u_, grad(phi))*C*theta_ + dot(K/Pr*grad(theta_), grad(phi)) 
        - dot(u_n, v)/dt
        - C*theta_n*phi/dt
        )*fenics.dx
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w_, dw)
        
    else:

        JF = (
            b(du, q) - gamma*dp*q
            + dot(du, v)/dt + c(u_, du, v) + c(du, u_, v) + a(dtheta*ddtheta_mu_sl, u_, v) + a(mu_sl, du, v) + b(v, dp)
            + dot(dtheta*ddtheta_m_B*g, v)
            + C*dtheta*phi/dt - dot(du, grad(phi))*C*theta_ - dot(u_, grad(phi))*C*dtheta + K/Pr*dot(grad(dtheta), grad(phi))
            )*fenics.dx
    
    
    # Solve nonlinear problem.
    problem = fenics.NonlinearVariationalProblem(F, w_, bcs, JF)

    solver  = fenics.NonlinearVariationalSolver(problem)

    solver.solve()
        
        
    # Write the solution.
    velocity, pressure, temperature = w_.split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("theta", "temperature")
    
    output_dir = 'output/nonlinear_mixedfe_unsteady_energy_autoJ'+str(automatic_jacobian)
    
    solution_files = [fenics.XDMFFile(output_dir + '/velocity.xdmf'), fenics.XDMFFile(output_dir + '/pressure.xdmf'),
        fenics.XDMFFile(output_dir + '/temperature.xdmf')]
        
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_files[i].write(var)



if __name__=='__main__':
    
    nonlinear_mixedfe(automatic_jacobian=True)
    
    nonlinear_mixedfe(automatic_jacobian=False)
    