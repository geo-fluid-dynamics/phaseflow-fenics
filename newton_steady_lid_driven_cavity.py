''' Solve the steady lid-driven cavity flow benchmark problem 

    modeled by the steady Navier-Stokes equations
    
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
    output_dir = 'output_steady_nldc/', \
    gamma = 1.e-7, \
    mesh_M = 80, \
    pressure_order = 1, \
    linearize = False, \
    newton_absolute_tolerance = 1.e-8, \
    ):

    dim = 2
    
    mu = 1.
    
    Re = 1.
    
    # Compute derived parameters
    velocity_order = pressure_order + 1
    

    # Create mesh
    mesh = UnitSquareMesh(mesh_M, mesh_M)


    # Define function spaces for the system
    VxV = VectorFunctionSpace(mesh, 'P', velocity_order)

    Q = FunctionSpace(mesh, 'P', pressure_order) # @todo mixing up test function space

    '''
    MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    VxV_ele = VectorElement('P', mesh.ufl_cell(), velocity_order)

    '''
    @todo How can we use the space $Q = \left{q \in L^2(\Omega) | \int{q = 0}\right}$ ?
    
    All Navier-Stokes FEniCS examples I've found simply use P2P1. danaila2014newton says that
    they are using the "classical Hilbert spaces" for velocity and pressure, but then they write
    down the space Q with less restrictions than H^1_0.
    
    '''
    Q_ele = FiniteElement('P', mesh.ufl_cell(), pressure_order)

    W_ele = MixedElement([VxV_ele, Q_ele])

    W = FunctionSpace(mesh, W_ele)


    # Define function and test functions
    w = Function(W)   

    v, q = TestFunctions(W)

        
    # Split solution function to access variables separately
    u, p = split(w)

    
    # Define boundary conditions
    lid = 'near(x[1],  1.)'
    
    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'
    
    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'
    
    bc = [ \
        DirichletBC(W.sub(0), Constant((1., 0.)), lid), \
        DirichletBC(W.sub(0), Constant((0., 0.)), fixed_walls), \
        DirichletBC(W.sub(1), Constant(0.), bottom_left_corner, method='pointwise')]

    
    # Define expressions needed for variational form
    Re = Constant(Re)

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
            + c(u, u, v) + a(mu, u, v) + b(v, p) \
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
        
        w_k = project(Constant((0., 0., 0.)), W)
        
        u_k, p_k = split(w_k)
        
        A = (\
            b(u_w,q) - gamma*p_w*q \
            + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu, u_w, v) + b(v, p_w) \
            )*dx
            
        L = (\
            b(u_k,q) + gamma*p_k*q \
            + c(u_k, u_k, v) + a(mu, u_k, v) + b(v, p_k) \
            )*dx  


    # Define method for writing values, and write initial values# Create VTK file for visualization output
    velocity_file = File(output_dir + '/velocity.pvd')

    pressure_file = File(output_dir + '/pressure.pvd')       
        
    w_w = Function(W) # w_w was previously a TrialFunction, but must be a Function when calling solve()
        
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
    _velocity, _pressure = w.split()
    
    velocity_file << _velocity
    
    pressure_file << _pressure
    
    
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()