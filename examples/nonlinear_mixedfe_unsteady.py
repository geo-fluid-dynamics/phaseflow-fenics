import fenics


''' @brief Example of nonlinear problem with mixed finite elements

@detail This is the well-known steady lid-driven cavity problem,
modeled by the incompressible Navier-Stokes mass and momentum equations.

Here we attempt to use the scripts from section 1.2.4 of the FEniCS book.

@todo Debug the Jacobian bilinear variational form.'''

def nonlinear_mixedfe(automatic_jacobian=True):

    # Set physical parameters
    mu = 0.01
    
    
    # Set numerical parameters.
    mesh = fenics.UnitSquareMesh(10, 10, 'crossed')
    
    gamma = 1.e-7
    
    pressure_degree = 1

    dt = 1.e12
    

    # Set function spaces for the variational form     .
    velocity_degree = pressure_degree + 1
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0 '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    W_ele = fenics.MixedElement([velocity_element, pressure_element])

    W = fenics.FunctionSpace(mesh, W_ele)  
    
    
    # Set initial values and boundary conditions.
    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'
    
    w_n = fenics.interpolate(fenics.Expression((lid, "0.", "0."), element=W_ele), W)
                    
    bcs = [
        fenics.DirichletBC(W.sub(0), fenics.Expression(("1.", "0."), degree=velocity_degree + 1),
            lid, method='topological'),
        fenics.DirichletBC(W.sub(0), fenics.Expression(("0.", "0."), degree=velocity_degree + 1),
            fixed_walls, method='topological'),
        fenics.DirichletBC(W.sub(1), fenics.Expression("0.", degree=pressure_degree + 1),
            bottom_left_corner, method='pointwise')]    
    
    
    # Set nonlinear variational form.
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    
    # Solve nonlinear problem.
    u_n, p_n = fenics.split(w_n)
    
    dw = fenics.TrialFunction(W)
    
    du, dp = fenics.split(dw)
    
    v, q = fenics.TestFunctions(W)
        
    w_ = fenics.Function(W)
    
    u_, p_ = fenics.split(w_)
    
    F = (b(u_, q) - gamma*p_*q + dot(u_, v)/dt + c(u_, u_, v) + a(mu, u_, v) + b(v, p_) - dot(u_n, v)/dt)*fenics.dx
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w_, dw)
        
    else:

        JF = (b(du, q) - gamma*dp*q + dot(du, v)/dt + c(u_, du, v) + c(du, u_, v) + a(mu, du, v) + b(v, dp))*fenics.dx
    
    problem = fenics.NonlinearVariationalProblem(F, w_, bcs, JF)

    solver  = fenics.NonlinearVariationalSolver(problem)

    solver.solve()
        
    velocity, pressure = w_.split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    output_dir = 'output/nonlinear_mixedfe_unsteady_autoJ'+str(automatic_jacobian)
    
    solution_files = [fenics.XDMFFile(output_dir + '/velocity.xdmf'), fenics.XDMFFile(output_dir + '/pressure.xdmf')]
        
    for i, var in enumerate([velocity, pressure]):
    
        solution_files[i].write(var)



if __name__=='__main__':
    
    nonlinear_mixedfe(automatic_jacobian=True)
    
    nonlinear_mixedfe(automatic_jacobian=False)
    