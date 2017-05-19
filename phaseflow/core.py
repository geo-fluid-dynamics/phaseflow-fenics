''' Solve the benchmark "differentially heated cavity" natural convection problem using finite elements.
        
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
    
'''
import fenics   
import dolfin
import helpers
import time


'''
Per Equation 8 from danaila2014newton, we implement an equation scaled with v_ref = nu_liquid/H => t_ref = nu_liquid/H^2 => Re = 1.
''' 
REYNOLDS_NUMBER = 1.

DEFAULT_RAYLEIGH_NUMBER = 1.e6
    
DEFAULT_PRANDTL_NUMBER = 0.71

DEFAULT_STEFAN_NUMBER = 0.045

            
'''@todo First add variable viscosity, later latent heat source term.
Conceptually this will be like having a PCM with zero latent heat.
The melting front should move quickly.'''

def function_spaces(mesh=fenics.UnitSquareMesh(20, 20, "crossed"), pressure_degree=1, temperature_degree=1):
    """ Define function spaces for the variational form """
    
    velocity_degree = pressure_degree + 1
    
    VxV = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    Q = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    V = fenics.FunctionSpace(mesh, 'P', temperature_degree)

    '''
    MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    VxV_ele = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    '''
    @todo How can we use the space $Q = \left{q \in L^2(\Omega) | \int{q = 0}\right}$ ?

    All Navier-Stokes FEniCS examples I've found simply use P2P1. danaila2014newton says that
    they are using the "classical Hilbert spaces" for velocity and pressure, but then they write
    down the space Q with less restrictions than H^1_0.

    '''
    Q_ele = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    V_ele = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    W_ele = fenics.MixedElement([VxV_ele, Q_ele, V_ele])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    return W, W_ele
    
    
def run(
    output_dir = "output/natural_convection",
    Ra = DEFAULT_RAYLEIGH_NUMBER,
    Pr = DEFAULT_PRANDTL_NUMBER,
    K = 1.,
    mu_l = 1.,
    g = (0., -1.),
    m_B = lambda theta : theta*DEFAULT_RAYLEIGH_NUMBER/(DEFAULT_PRANDTL_NUMBER*REYNOLDS_NUMBER**2),
    dm_B_dtheta = lambda theta : DEFAULT_RAYLEIGH_NUMBER/(DEFAULT_PRANDTL_NUMBER*REYNOLDS_NUMBER**2),
    mesh=fenics.UnitSquareMesh(20, 20, "crossed"),
    initial_values_expression = (
        "0.",
        "0.",
        "0.",
        "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
    bc_expressions = [
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"],
        [2, "0.", 2, "near(x[0],  0.)", "topological"],
        [2, "0.", 2, "near(x[0],  1.)", "topological"]],
    final_time = 1.,
    time_step_bounds = (1.e-3, 1.e-3, 1.),
    adaptive_space = False,
    adaptive_space_error_tolerance = 1.e-4,
    gamma = 1.e-7,
    pressure_degree = 1,
    temperature_degree = 1,
    linearize = True,
    newton_relative_tolerance = 1.e-9,
    max_newton_iterations = 10,
    stop_when_steady = True,
    steady_relative_tolerance = 1.e-8):

    #
    print("Running Phaseflow with the following arguments:")
    
    print(helpers.arguments())
    
    # Validate inputs    
    if type(time_step_bounds) == type(1.):
    
        time_step_bounds = (time_step_bounds, time_step_bounds, time_step_bounds)
    
    time_step_size = time.TimeStepSize(helpers.BoundedValue(time_step_bounds[0], time_step_bounds[1], time_step_bounds[2]))
        
    
    # Compute derived parameters
    W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
        
        
    # Define function and test functions
    w = fenics.Function(W)   

    v, q, phi = fenics.TestFunctions(W)

        
    # Split solution function to access variables separately
    u, p, theta = fenics.split(w)

       
    # Specify the initial values
    w_n = fenics.interpolate(fenics.Expression(initial_values_expression, element=W_ele), W)
    
    u_n, p_n, theta_n = fenics.split(w_n)

    
    # Define expressions needed for variational form
    Ra = fenics.Constant(Ra)

    Pr = fenics.Constant(Pr)

    Re = fenics.Constant(REYNOLDS_NUMBER)

    K = fenics.Constant(K)

    g = fenics.Constant(g)

    gamma = fenics.Constant(gamma)
    
    mu_l = fenics.Constant(mu_l)


    # Define variational form
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    def a(_mu, _u, _v):

        def D(_u):
        
            return sym(grad(_u))
        
        return 2.*_mu*inner(D(_u), D(_v))
        

    def b(_u, _q):
        
        return -div(_u)*_q
        

    def c(_w, _z, _v):
       
        return dot(dot(grad(_z), _w), _v)
    
    # Specify boundary conditions
    bc = []
    
    for subspace, expression, degree, coordinates, method in bc_expressions:
    
        bc.append(fenics.DirichletBC(W.sub(subspace), fenics.Expression(expression, degree=degree), coordinates, method=method))
        

    # Implement the nonlinear variational form
    if not linearize:
        
        def nonlinear_variational_form(dt, w_n):
        
            u_n, p_n, theta_n = fenics.split(w_n)
        
            dt = fenics.Constant(dt)
            
            F = (\
                    b(u, q) - gamma*p*q
                    + dot(u, v)/dt + c(u, u, v) + a(mu_l, u, v) + b(v, p) - dot(u_n, v)/dt + dot(m_B(theta)*g, v)
                    + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt
                    )*fenics.dx
            
            return F
            
            
    # Implement the Newton linearized form published in danaila2014newton
    elif linearize: 

        w_w = fenics.TrialFunction(W)
        
        u_w, p_w, theta_w = fenics.split(w_w)
        
        w_k = fenics.Function(W)
        
        w_k.assign(w_n)
    
        u_k, p_k, theta_k = fenics.split(w_k)

        def linear_variational_form(dt, w_k):
        
            dt = fenics.Constant(dt)
        
            u_k, p_k, theta_k = fenics.split(w_k)
            
            A = (\
                b(u_w, q) - gamma*p_w*q
                + dot(u_w, v)/dt + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu_l, u_w, v) + b(v, p_w) + dot(dm_B_dtheta(theta_k)*theta_w*g, v)
                + theta_w*phi/dt - dot(u_k, grad(phi))*theta_w - dot(u_w, grad(phi))*theta_k + dot(K/Pr*grad(theta_w), grad(phi))
                )*fenics.dx
                
            L = (\
                b(u_k, q) - gamma*p_k*q
                + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu_l, u_k, v) + b(v, p_k) + dot(m_B(theta_k)*g, v)
                + (theta_k - theta_n)*phi/dt - dot(u_k, grad(phi))*theta_k + dot(K/Pr*grad(theta_k), grad(phi))
                )*fenics.dx  
                
            return A, L


    # Create progress bar
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
    

    # Define method for writing values, and write initial values# Create VTK file for visualization output
    solution_files = [fenics.File(output_dir + '/velocity.pvd'), fenics.File(output_dir + '/pressure.pvd'), fenics.File(output_dir + '/temperature.pvd')]

    def write_solution(solution_files, _w, current_time):

        w = _w.leaf_node()
        
        velocity, pressure, temperature = w.split()
        
        velocity.rename("u", "velocity")
        
        pressure.rename("p", "pressure")
        
        temperature.rename("theta", "temperature")
        
        for i, var in enumerate([velocity, pressure, temperature]):
        
            solution_files[i] << (var, current_time) 


    current_time = 0.
    
    write_solution(solution_files, w_n, current_time) 

    
    # Solve each time step
    time_residual = fenics.Function(W)
    
    def solve_time_step(dt, w_n):
    
        if linearize:
        
            print '\nIterating Newton method'
            
            converged = False
            
            iteration_count = 0
        
            w_k.assign(w_n)
            
            for k in range(max_newton_iterations):
            
                A, L = linear_variational_form(dt, w_k)
                
                w_w = fenics.Function(W)

                u_w, p_w, theta_w = fenics.split(w_w.leaf_node())

                if not adaptive_space:
                
                    fenics.solve(A == L, w_w, bcs=bc)
                    
                else:
                        
                    '''w_w was previously a TrialFunction, but must be a Function when defining M and when calling solve().
                    This details here are opaque to me. Here is a related issue: https://fenicsproject.org/qa/12271/adaptive-stokes-perform-compilation-unable-extract-indices'''
                    M = fenics.sqrt((u_k[0] - u_w[0])**2 + (u_k[1] - u_w[1])**2 + (theta_k - theta_w)**2)*fenics.dx
                        
                    problem = fenics.LinearVariationalProblem(A, L, w_w, bcs=bc)

                    solver = fenics.AdaptiveLinearVariationalSolver(problem, M)

                    solver.solve(adaptive_space_error_tolerance)

                    solver.summary()
    
                w_k.assign(w_k - w_w)
                
                norm_residual = fenics.norm(w_w, 'L2')/fenics.norm(w_k, 'L2')

                print '\nL2 norm of relative residual, || w_w || / || w_k || = ' + str(norm_residual) + '\n'
                
                if norm_residual < newton_relative_tolerance:
                    
                    iteration_count = k + 1
                    
                    print 'Converged after ' + str(k) + ' iterations'
                    
                    converged = True
                    
                    break
                     
            w_out = w_k # Here we can't simply say w.assign(w_k), because w was not refined (because it had no relation to the adaptive problem)

        else:
        
            '''  @todo Implement adaptive time for nonlinear version.
            How to get residual from solver.solve() to check if diverging? 
            Related: Set solver.parameters.nonlinear_variational_solver.newton_solver["error_on_nonconvergence"] = False and figure out how to read convergence data.'''
            
            F = nonlinear_variational_form(dt, w_n)
            
            problem = fenics.NonlinearVariationalProblem(F, w, bc, fenics.derivative(F, w))
            
            if not adaptive_space:     
            
                solver = fenics.NonlinearVariationalSolver(problem)
                
                iteration_count, converged = solver.solve()
                
            else:
        
                M = fenics.sqrt(u[0]**2 + u[1]**2 + theta**2)*fenics.dx
            
                solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)

                solver.solve(adaptive_space_error_tolerance)

                solver.summary()
                
                converged = True 
                
            assert(converged)
            
            w_out = w
            
        return w_out, converged

    EPSILON = 1.e-12

    while current_time < final_time - EPSILON:

        remaining_time = final_time - current_time
    
        if time_step_size.value > remaining_time:
            
            time_step_size.set(remaining_time)
    
        converged = False
        
        while not converged:
        
            w, converged = solve_time_step(time_step_size.value, w_n)
            
            if time_step_size.value <= time_step_size.min + dolfin.DOLFIN_EPS:
                    
                break;
            
            if not converged:
            
                time_step_size.set(time_step_size.value/2.)
    
        current_time += time_step_size.value
        
        ''' Save solution to files. Saving here allows us to inspect the latest solution 
        even if the Newton iterations failed to converge.'''
        write_solution(solution_files, w, current_time)
        
        assert(converged)
        
        time_step_size.set(2*time_step_size.value)
            
        print 'Reached time t = ' + str(current_time)
        
        if stop_when_steady:
        
            # Check for steady state
            time_residual.assign(w - w_n)
        
        # Update previous solution
        w_n.assign(w) # We cannot simply use w_n.assign(w), because w may have been refined
        
        # Show the time progress
        progress.update(current_time / final_time)
        
        if stop_when_steady:
        
            unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
            
            print 'Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = ' + str(unsteadiness)
        
            if (unsteadiness < steady_relative_tolerance):
                print 'Reached steady state at time t = ' + str(current_time)
                break
    
    if time >= final_time:
    
        print 'Reached final time, t = ' + str(final_time)
        
        
    w_n.rename("w", "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
        
    return fe_field_interpolant
    
    
if __name__=='__main__':

    run()