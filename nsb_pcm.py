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
        
        Match the notation in danaila2014newton as best as possible.

    
    @author Alexander G. Zimmerman <zimmerman@aices.rwth-aachen.de>
    
'''

from fenics import \
    UnitSquareMesh, FiniteElement, VectorElement, MixedElement, \
    FunctionSpace, VectorFunctionSpace, \
    Function, TrialFunction, TestFunctions, split, \
    DirichletBC, Constant, Expression, \
    dx, \
    dot, inner, grad, nabla_grad, sym, div, \
    errornorm, norm, \
    File, \
    Progress, set_log_level, PROGRESS, \
    project, interpolate, \
    solve, parameters, info, derivative, \
    LinearVariationalProblem, LinearVariationalSolver, NonlinearVariationalProblem, NonlinearVariationalSolver, \
    SubDomain, EdgeFunction, near, adapt


def arguments():
    """Returns tuple containing dictionary of calling function's
       named arguments and a list of calling function's unnamed
       positional arguments.
    """
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    return args, posargs

    
def run(
    output_dir = 'output_nsb_pcm', \
    Ra = 1.e6, \
    Pr = 0.71, \
    K = 1., \
    g = (0., -1.), \
    mu = 1., \
    mesh = UnitSquareMesh(20, 20, "crossed"), \
    s_u = ('0.', '0.'), \
    s_p = '0.', \
    s_theta = '0.', \
    initial_values_expression = ( \
        '0.', \
        '0.', \
        '0.', \
        '0.5*near(x[0],  0.) -0.5*near(x[0],  1.)'), \
    bc_expressions = [ \
        [0, ('0.', '0.'), 3, 'near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)'], \
        [2, '0.', 2, 'near(x[0],  0.)'], \
        [2, '0.', 2, 'near(x[0],  1.)']], \
    final_time = 1., \
    time_step_size = 1.e-3, \
    adaptive_time = True, \
    gamma = 1.e-7, \
    pressure_degree = 1, \
    temperature_degree = 1, \
    linearize = True, \
    newton_relative_tolerance = 1.e-9, \
    max_newton_iterations = 10, \
    stop_when_steady = True, \
    steady_relative_tolerance = 1.e-8, \
    debug_b_factor = 1., \
    debug_c_factor = 1., \
    exact_solution_expression = [] \
    ):

    #
    print("Running nsb_pcm with the following arguments:")
    
    print(arguments())
    
    # @todo Try 3D
    dim = 2
    
    '''
    Per Equation 8 from danaila2014newton, we implement an equation scaled with v_ref = nu_liquid/H => t_ref = nu_liquid/H^2 => Re = 1.
    ''' 
    Re = 1.
    
    # Compute derived parameters
    velocity_degree = pressure_degree + 1
    

    # Define function spaces for the system
    VxV = VectorFunctionSpace(mesh, 'P', velocity_degree)

    Q = FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    V = FunctionSpace(mesh, 'P', temperature_degree)

    '''
    MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    VxV_ele = VectorElement('P', mesh.ufl_cell(), velocity_degree)

    '''
    @todo How can we use the space $Q = \left{q \in L^2(\Omega) | \int{q = 0}\right}$ ?

    All Navier-Stokes FEniCS examples I've found simply use P2P1. danaila2014newton says that
    they are using the "classical Hilbert spaces" for velocity and pressure, but then they write
    down the space Q with less restrictions than H^1_0.

    '''
    Q_ele = FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    V_ele = FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    W_ele = MixedElement([VxV_ele, Q_ele, V_ele])

    W = FunctionSpace(mesh, W_ele)    
        
    # Define function and test functions
    w = Function(W)   

    v, q, phi = TestFunctions(W)

        
    # Split solution function to access variables separately
    u, p, theta = split(w)
    
    
    # Set source term expressions
    s_u = Expression(s_u, element=VxV_ele, mu=mu, gamma=gamma)
    
    s_p = Expression(s_p, element=Q_ele, mu=mu, gamma=gamma)
    
    s_theta = Expression(s_theta, element=V_ele)
        
       
    # Specify the initial values
    w_n = interpolate(Expression(initial_values_expression, element=W_ele), W)
    
    u_n, p_n, theta_n = split(w_n)

    
    # Define expressions needed for variational form
    Ra = Constant(Ra)

    Pr = Constant(Pr)

    Re = Constant(Re)

    K = Constant(K)

    mu = Constant(mu)

    g = Constant(g)

    gamma = Constant(gamma)


    # Define variational form
    def f_B(_theta):
        
        return _theta*Ra/(Pr*Re*Re)*g
           
       
    def a(_mu, _u, _v):

        def D(_u):
        
            return sym(grad(_u))
        
        return 2.*_mu*inner(D(_u), D(_v))
        

    def b(_u, _q):
        
        return -div(_u)*_q*debug_b_factor
        

    def c(_w, _z, _v):
       
        return dot(dot(_w, nabla_grad(_z)), _v)*debug_c_factor # @todo Is this use of nabla_grad correct?
    
    # Specify boundary conditions
    bc = []
    
    for subspace, expression, degree, coordinates, method in bc_expressions:
    
        bc.append(DirichletBC(W.sub(subspace), Expression(expression, degree=degree), coordinates, method=method))
        

    # Implement the nonlinear variational form
    if not linearize:
        
        def nonlinear_variational_form(dt):
        
            dt = Constant(dt)
            
            F = (\
                    b(u, q) - gamma*p*q - s_p*q \
                    + dot(u, v)/dt + c(u, u, v) + a(mu, u, v) + b(v, p) - dot(u_n, v)/dt + dot(f_B(theta), v) - dot(s_u, v) \
                    + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt - s_theta*phi \
                    )*dx
            
            return F
    # Implement the Newton linearized form published in danaila2014newton
    elif linearize: 

        w_w = TrialFunction(W)
        
        u_w, p_w, theta_w = split(w_w)
        
        w_k = Function(W)
        
        w_k.assign(w_n)
    
        u_k, p_k, theta_k = split(w_k)

        df_B_dtheta = Ra/(Pr*Re*Re)*g

        def linear_variational_form(dt):
        
            dt = Constant(dt)
        
            A = (\
                b(u_w,q) - gamma*p_w*q \
                + dot(u_w, v)/dt + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu, u_w, v) + b(v, p_w) + dot(theta_w*df_B_dtheta, v) \
                + theta_w*phi/dt - dot(u_k, grad(phi))*theta_w - dot(u_w, grad(phi))*theta_k + dot(K/Pr*grad(theta_w), grad(phi)) \
                )*dx
                
            L = (\
                b(u_k,q) - gamma*p_k*q + s_p*q \
                + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu, u_k, v) + b(v, p_k) + dot(f_B(theta_k), v) + dot(s_u, v) \
                + (theta_k - theta_n)*phi/dt - dot(u_k, grad(phi))*theta_k + dot(K/Pr*grad(theta_k), grad(phi)) + s_theta*phi \
                )*dx  
                
            return A, L


    # Create progress bar
    progress = Progress('Time-stepping')

    set_log_level(PROGRESS)
    

    # Define method for writing values, and write initial values# Create VTK file for visualization output
    velocity_file = File(output_dir + '/velocity.pvd')

    pressure_file = File(output_dir + '/pressure.pvd')

    temperature_file = File(output_dir + '/temperature.pvd')

    def write_solution(_w, time):

        _velocity, _pressure, _temperature = _w.split()
        
        _velocity.rename("u", "velocity")
        
        _pressure.rename("p", "pressure")
        
        _temperature.rename("theta", "temperature")
        
        velocity_file << (_velocity, time) 
        
        pressure_file << (_pressure, time) 
        
        temperature_file << (_temperature, time) 


    time = 0.

    w.assign(w_n)
    
    write_solution(w, time) 

    # Solve each time step
    
    w_w = Function(W) # w_w was previously a TrialFunction, but must be a Function when calling solve()

    time_residual = Function(W)
    
    def solve_time_step(dt):
    
        if linearize:
        
            print '\nIterating Newton method'
            
            converged = False
            
            iteration_count = 0
            
            old_residual = 1e32
        
            w_k.assign(w_n)
            
            for k in range(max_newton_iterations):
            
                A, L = linear_variational_form(dt)

                solve(A == L, w_w, bcs=bc)
                
                w_k.assign(w_k - w_w)
                
                norm_residual = norm(w_w, 'L2')/norm(w_k, 'L2')

                print '\nL2 norm of relative residual, || w_w || / || w_k || = ' + str(norm_residual) + '\n'
                
                if norm_residual > old_residual:
                
                    diverging = True
                    
                    if not adaptive_time:
                    
                        assert(not diverging)
                    
                    return diverging
                
                old_residual = norm_residual
                
                if norm_residual < newton_relative_tolerance:
                
                    converged = True
                    
                    iteration_count = k + 1
                    
                    print 'Converged after ' + str(k) + ' iterations'
                    
                    break
                    
            assert(converged)
            
            w.assign(w_k)
            
            diverging = False
            
            return diverging

        else:
        
            assert(not adaptive_time) # @todo How to get residual from solver.solve() to check if diverging?
        
            F = nonlinear_variational_form(dt)
        
            problem = NonlinearVariationalProblem(F, w, bc, derivative(F, w))
            
            solver = NonlinearVariationalSolver(problem)
            
            iteration_count, converged = solver.solve()
            
            assert(converged)

    EPSILON = 1.e-12

    while time < final_time - EPSILON:

        if adaptive_time:
    
            remaining_time = final_time - time
        
            if time_step_size > remaining_time:
                
                time_step_size = remaining_time        
        
            diverging = True
            
            while diverging:
            
                diverging = solve_time_step(time_step_size)
                
                if diverging:
                
                    # @todo Expose parameters for maximum and minimum size.
                
                    time_step_size /= 2.
                    
                    print 'Set time step size to dt = ' + str(time_step_size)
        
            time += time_step_size
            
            time_step_size *= 2.
            
            print 'Set time step size to dt = ' + str(time_step_size)

        else:
            
            solve_time_step(time_step_size)
            
            time += time_step_size
            
        print 'Reached time t = ' + str(time)
            
        # Save solution to files
        write_solution(w, time)
        
        if stop_when_steady:
        
            # Check for steady state
            time_residual.assign(w - w_n)
        
        # Update previous solution
        w_n.assign(w)
        
        # Show the time progress
        progress.update(time / final_time)
        
        if stop_when_steady:
        
            unsteadiness = norm(time_residual, 'L2')/norm(w_n, 'L2')
            
            print 'Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = ' + str(unsteadiness)
        
            if (unsteadiness < steady_relative_tolerance):
                print 'Reached steady state at time t = ' + str(time)
                break
    
    if time >= final_time:
    
        print 'Reached final time, t = ' + str(final_time)
        
    
    if exact_solution_expression:
    
        w_e = Expression(exact_solution_expression, element=W_ele)
    
        error = errornorm(w_e, w_n, 'L2')
    
        print("Error = " + str(error))
    
    
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()