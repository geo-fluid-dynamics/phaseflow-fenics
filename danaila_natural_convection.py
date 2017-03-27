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
    dot, inner, grad, sym, div, \
    errornorm, norm, \
    File, \
    Progress, set_log_level, PROGRESS, \
    project, interpolate, \
    solve, parameters, info


def run(
    Ra = 1.e6, \
    Pr = 0.71, \
    Re = 1., \
    theta_h = 0.5, \
    theta_c = -0.5, \
    K = 1., \
    g = (0., -1.), \
    mu = 1., \
    final_time = 1., \
    time_step_size = 1.e-3, \
    gamma = 1.e-7, \
    mesh_M = 80, \
    pressure_order = 1, \
    temperature_order = 1, \
    linearize = True, \
    newton_absolute_tolerance = 1.e-12, \
    stop_when_steady = True, \
    steady_absolute_tolerance = 1.e-12 \
    ):

    # Compute derived parameters
    velocity_order = pressure_order + 1
    

    # Create mesh
    mesh = UnitSquareMesh(mesh_M, mesh_M)


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

    W_ele = MixedElement([VxV_ele, Q_ele, V_ele])

    W = FunctionSpace(mesh, W_ele)


    # Define function and test functions
    w = Function(W)   

    v, q, phi = TestFunctions(W)

        
    # Split solution function to access variables separately
    u, p, theta = split(w)
       

    # Define boundaries
    hot_wall = 'near(x[0],  0.)'

    cold_wall = 'near(x[0],  1.)'

    adiabatic_walls = 'near(x[1],  0.) | near(x[1],  1.)'


    # Define boundary conditions
    bc = [ \
        DirichletBC(W, Constant((0., 0., 0., theta_h)), hot_wall), \
        DirichletBC(W, Constant((0., 0., 0., theta_c)), cold_wall), \
        DirichletBC(W.sub(0), Constant((0., 0.)), adiabatic_walls), \
        DirichletBC(W.sub(1), Constant(0.), adiabatic_walls)]
        

    # Specify the initial values
    initial_values = Expression( \
            ('0.', '0.', '0.', 'near(x[0],  0.)*' + str(theta_h) + '  + near(x[0], 1.)*' + str(theta_c)),
            element=W_ele)
                
    w_n = project(initial_values, W)

    u_n, p_n, theta_n = split(w_n)


    # Define expressions needed for variational form
    Ra = Constant(Ra)

    Pr = Constant(Pr)

    Re = Constant(Re)

    K = Constant(K)

    mu = Constant(mu)

    g = Constant(g)

    dt = Constant(time_step_size)

    gamma = Constant(gamma)


    # Define variational form
    def f_B(_theta):
        
        return _theta*Ra/(Pr*Re*Re)*g
           
       
    def a(_mu, _u, _v):

        def D(_u):
        
            return sym(grad(_u))
        
        return 2.*_mu*inner(D(_u), D(_v))
        

    def b(_u, _q):
        
        return -div(_u)*_q
        

    def c(_w, _z, _v):
        
        return dot(div(_w)*_z, _v)
        
        
    # Implement the nonlinear form, which will allow FEniCS to automatically derive the Newton linearized form.
    F = (\
            b(u, q) - gamma*p*q \
            + dot(u, v)/dt + c(u, u, v) + a(mu, u, v) + b(v, p) - dot(u_n, v)/dt \
            - dot(f_B(theta), v) \
            + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt \
            )*dx


    # Implement the Newton linearized form published in danaila2014newton
    if linearize: 

        max_newton_iterations = 10

        w_w = TrialFunction(W)
        
        u_w, p_w, theta_w = split(w_w)

        def boundary(x, on_boundary):
            return on_boundary
        
        bc_dot = DirichletBC(W, Constant((0., 0., 0., 0.)), boundary)
        
        w_k = project(Constant((0., 0., 0., 0.)), W)
        
        u_k, p_k, theta_k = split(w_k)

        df_B_dtheta = Ra/(Pr*Re*Re)*g
        
        A = (\
            b(u_w,q) - gamma*p_w*q \
            + dot(u_w, v)/dt + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu, u_w, v) + b(v, p_w) \
            - dot(theta_w*df_B_dtheta, v) \
            + theta_w*phi/dt - dot(u_k, grad(phi))*theta_w - dot(u_w, grad(phi))*theta_k + dot(K/Pr*grad(theta_w), grad(phi)) \
            )*dx
            
        L = (\
            b(u_k,q) + gamma*p_k*q \
            + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu, u_k, v) + b(v, p_k) - dot(f_B(theta_k), v) \
            + (theta_k - theta_n)*phi/dt - dot(u_k, grad(phi))*theta_k - dot(K/Pr*grad(theta_k), grad(phi)) \
            )*dx  


    # Create progress bar
    progress = Progress('Time-stepping')

    set_log_level(PROGRESS)


    # solve() requires the second argument to be a Function instead of a TrialFunction
    _w_w = Function(W)

    # Define method for writing values, and write initial values# Create VTK file for visualization output
    output_dir = 'output_danaila_natural_convection'

    velocity_file = File(output_dir + '/velocity.pvd')

    pressure_file = File(output_dir + '/pressure.pvd')

    temperature_file = File(output_dir + '/temperature.pvd')

    def write_solution(_w, time):

        _velocity, _pressure, _temperature = _w.split()
        
        velocity_file << (_velocity, time) 
        
        pressure_file << (_pressure, time) 
        
        temperature_file << (_temperature, time) 


    time = 0.

    write_solution(w_n, time) 

    # Solve each time step

    time_residual = Function(W)

    while time < final_time:

        time += time_step_size
        
        if linearize:
        
            print '\nIterating Newton method'
            
            converged = False
            
            iteration_count = 0
            
            old_residual = 1e32
            
            for k in range(max_newton_iterations):

                solve(A == L, _w_w, bc_dot)
                
                w_k.assign(w_k - _w_w)
                
                norm_residual = norm(_w_w, 'H1')

                print '\nH1 norm residual = ' + str(norm_residual) + '\n'
                
                assert(norm_residual < old_residual)
                
                old_residual = norm_residual
                
                if norm_residual < newton_absolute_tolerance:
                
                    converged = True
                    
                    iteration_count = k + 1
                    
                    print 'Converged after ' + str(k) + ' iterations'
                    
                    break
                    
            assert(converged)
            
            w.assign(w_k)

        else:
        
            solve(F == 0, w, bc)
        
        
        # Save solution to files
        write_solution(w, time)
        
        if stop_when_steady:
            # Check for steady state
            time_residual.assign(w - w_n)
        
        # Update previous solution
        w_n.assign(w)
        
        # Show the time progress
        progress.update(time / final_time)
        
        if stop_when_steady & (norm(time_residual, 'H1') < steady_absolute_tolerance):
            print 'Reached steady state at time t = ' + str(time)
            break
    
    
def test():

    danaila_natural_convection()
    
    pass
    
    
if __name__=='__main__':

    test()