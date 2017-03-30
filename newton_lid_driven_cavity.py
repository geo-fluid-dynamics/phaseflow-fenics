''' Solve the lid-driven cavity flow benchmark problem using the Newton-linearized Navier-Stokes equations.
        
    Use the Newton linearized mass and momentum equations from 

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
    
    This is being done to verify the mass and momentum parts of the system before solving the natural convection problem, which also couples the energy equation.
    
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
    solve, parameters, info


def run(
    output_dir = 'output_nldc/', \
    final_time = 1., \
    time_step_size = 1.e-1, \
    gamma = 1.e-7, \
    mesh_M = 40, \
    pressure_degree = 1, \
    linearize = False, \
    newton_absolute_tolerance = 1.e-8, \
    stop_when_steady = True, \
    steady_absolute_tolerance = 1.e-8 \
    ):

    dim = 2
    
    mu = 1.
    
    Re = 1.
    
    # Compute derived parameters
    velocity_degree = pressure_degree + 1
    

    # Create mesh
    mesh = UnitSquareMesh(mesh_M, mesh_M)


    # Define function spaces for the system
    VxV = VectorFunctionSpace(mesh, 'P', velocity_degree)

    Q = FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    VxV_ele = VectorElement('P', mesh.ufl_cell(), velocity_degree)

    Q_ele = FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = MixedElement([VxV_ele, Q_ele])

    W = FunctionSpace(mesh, W_ele)


    # Define function and test functions
    w = Function(W)   

    v, q = TestFunctions(W)

        
    # Split solution function to access variables separately
    u, p = split(w)
       
       
    # Specify boundary conditions
    lid = 'near(x[1],  1.)'
    
    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'
    
    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'
    
    bc = [ \
        DirichletBC(W.sub(0), Constant((1., 0.)), lid), \
        DirichletBC(W.sub(0), Constant((0., 0.)), fixed_walls), \
        DirichletBC(W.sub(1), Constant(0.), bottom_left_corner, method='pointwise')]
        
        
    # Specify the initial values
    w_n = interpolate(Expression((lid+'*1.', '0.', '0.'), degree=1), W)
        
    u_n, p_n = split(w_n)
    
    
    # Define expressions needed for variational form
    Re = Constant(Re)
    
    dt = Constant(time_step_size)

    gamma = Constant(gamma)


    # Define variational form       
    def a(_mu, _u, _v):

        def D(_u):
        
            return sym(grad(_u))
        
        return 2.*_mu*inner(D(_u), D(_v))
        

    def b(_u, _q):
        
        return -div(_u)*_q
        

    def c(_w, _z, _v):
       
        return dot(dot(_w, nabla_grad(_z)), _v)
        
        
    # Implement the nonlinear form, which will allow FEniCS to automatically derive the Newton linearized form.
    F = (\
        b(u, q) - gamma*p*q \
        + dot(u, v)/dt + c(u, u, v) + a(mu, u, v) + b(v, p) - dot(u_n, v)/dt \
        )*dx


    # Implement the Newton linearized form published in danaila2014newton
    if linearize: 

        max_newton_iterations = 10

        w_w = TrialFunction(W)
        
        u_w, p_w = split(w_w)

        def boundary(x, on_boundary):
            return on_boundary
        
        '''
        danaila2014newton sets homogeneous Dirichlet BC's on all residuals, including theta.
        I don't see how this could be consistent with Neumann BC's for the nonlinear problem.
        A solution for the Neumann BVP can't be constructed by adding a series of residuals which 
        use a homogeneous Dirichlet BC.
        '''
        bc_dot = DirichletBC(W, Constant((0., 0., 0.)), boundary)
        
        w_k = Function(W)
        
        w_k.assign(w_n)
        
        u_k, p_k = split(w_k)
        
        A = (\
            b(u_w,q) - gamma*p_w*q \
            + dot(u_w, v)/dt + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu, u_w, v) + b(v, p_w) \
            )*dx
            
        L = (\
            b(u_k,q) + gamma*p_k*q \
            + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu, u_k, v) + b(v, p_k) \
            )*dx  


    # Create progress bar
    progress = Progress('Time-stepping')

    set_log_level(PROGRESS)
    

    # Define method for writing values, and write initial values# Create VTK file for visualization output
    velocity_file = File(output_dir + '/velocity.pvd')

    pressure_file = File(output_dir + '/pressure.pvd')

    def write_solution(_w, time):

        _velocity, _pressure = _w.split()
        
        velocity_file << (_velocity, time) 
        
        pressure_file << (_pressure, time) 


    time = 0.

    w.assign(w_n)
    
    write_solution(w, time) 

    # Solve each time step
    
    w_w = Function(W) # w_w was previously a TrialFunction, but must be a Function when calling solve()

    time_residual = Function(W)

    while time < final_time:

        time += time_step_size
        
        if linearize:
        
            print '\nIterating Newton method'
            
            converged = False
            
            iteration_count = 0
            
            old_residual = 1e32
            
            for k in range(max_newton_iterations):

                solve(A == L, w_w, bc_dot)
                
                w_k.assign(w_k - w_w)
                
                norm_residual = norm(w_w, 'H1')

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
    
    if time >= final_time:
    
        print 'Reached final time, t = ' + str(final_time)
    
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()