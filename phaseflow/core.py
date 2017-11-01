"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import helpers
import globals
import default

TIME_EPS = 1.e-8

pressure_degree = 1

temperature_degree = 1

def make_mixed_fe(cell):
    """ Define the mixed finite element.
    MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    """
    velocity_degree = pressure_degree + 1
    
    velocity_element = fenics.VectorElement('P', cell, velocity_degree)
    
    pressure_element = fenics.FiniteElement('P', cell, pressure_degree)

    temperature_element = fenics.FiniteElement('P', cell, temperature_degree)

    mixed_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])
    
    return mixed_element

    
def write_solution(solution_file, w_k, time):
    """Write the solution to disk."""

    helpers.print_once("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w_k.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_file.write(var, time)
        

def steady(W, w, w_n, steady_relative_tolerance):
    """Check if solution has reached an approximately steady state."""
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
    
    helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady
    
    
def run(output_dir = 'output/wang2010_natural_convection_air',
        rayleigh_number = default.parameters['Ra'],
        prandtl_number = default.parameters['Pr'],
        stefan_number = default.parameters['Ste'],
        heat_capacity = default.parameters['C'],
        thermal_conductivity = default.parameters['K'],
        liquid_viscosity = default.parameters['mu_l'],
        solid_viscosity = default.parameters['mu_s'],
        gravity = default.parameters['g'],
        m_B = default.m_B,
        ddT_m_B = default.ddT_m_B,
        penalty_parameter = 1.e-7,
        regularization = default.regularization,
        mesh=default.mesh,
        initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
        boundary_conditions = [{'subspace': 0,
                'value_expression': ("0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': 'topological'},
            {'subspace': 2,
                'value_expression': "0.5", 'degree': 2, 
                'location_expression': "near(x[0],  0.)", 'method': 'topological'},
             {'subspace': 2,
                'value_expression': "-0.5", 'degree': 2, 
                'location_expression': "near(x[0],  1.)", 'method': 'topological'}],
        start_time = 0.,
        end_time = 10.,
        time_step_size = 1.e-3,
        stop_when_steady = True,
        steady_relative_tolerance=1.e-4,
        adaptive = False,
        adaptive_metric = 'all',
        adaptive_solver_tolerance = 1.e-4,
        nlp_absolute_tolerance = 1.e-8,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        restart = False,
        restart_filepath = ''):
    """Run Phaseflow.
    
    Phaseflow is configured entirely through the arguments in this run() function.
    
    See the tests and examples for demonstrations of how to use this.
    """
    
    
    # Report inputs.
    helpers.print_once("Running Phaseflow with the following arguments:")
    
    helpers.print_once(helpers.arguments())
    
    helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + '/arguments.txt', 'w')
        
        arguments_file.write(str(helpers.arguments()))

        arguments_file.close()
    
    
    # Check if 1D/2D/3D.
    dimensionality = mesh.type().dim()
    
    helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Initialize time.
    if restart:
    
        with h5py.File(restart_filepath, 'r') as h5:
            
            time = h5['t'].value
            
            assert(abs(time - start_time) < TIME_EPS)
    
    else:
    
        time = start_time
    
    
    # Define the mixed finite element and the solution function space.
    W_ele = make_mixed_fe(mesh.ufl_cell())
    
    W = fenics.FunctionSpace(mesh, W_ele)
    
    
    # Set the initial values.
    if restart:
            
        mesh = fenics.Mesh()
        
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
        
            h5.read(mesh, 'mesh', True)
        
        W_ele = make_mixed_fe(mesh.ufl_cell())
    
        W = fenics.FunctionSpace(mesh, W_ele)
    
        w_n = fenics.Function(W)
    
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
        
            h5.read(w_n, 'w')
    
    else:

        w_n = fenics.interpolate(fenics.Expression(initial_values_expression,
            element=W_ele), W)
            
        
    # Organize the boundary conditions.
    bcs = []
    
    for item in boundary_conditions:
    
        bcs.append(fenics.DirichletBC(W.sub(item['subspace']), item['value_expression'],
            item['location_expression'], method=item['method']))
    
    
    # Set the variational form.
    """Set local names for math operators to improve readability."""
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    """The linear, bilinear, and trilinear forms b, a, and c, follow the common notation 
    for applying the finite element method to the incompressible Navier-Stokes equations,
    e.g. from danaila2014newton and huerta2003fefluids.
    """
    b = lambda u, q : -div(u)*q  # Divergence
    
    D = lambda u : sym(grad(u))  # Symmetric part of velocity gradient
    
    a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)  # Convection of the velocity field
    
    dt = fenics.Constant(time_step_size)
    
    Ra = fenics.Constant(rayleigh_number), 
    
    Pr = fenics.Constant(prandtl_number)
    
    Ste = fenics.Constant(stefan_number)
    
    C = fenics.Constant(heat_capacity)
    
    K = fenics.Constant(thermal_conductivity)

    g = fenics.Constant(gravity)
    
    f_B = lambda T : m_B(T)*g  # Buoyancy force, $f = ma$
    
    gamma = fenics.Constant(penalty_parameter)
    
    T_f = fenics.Constant(regularization['T_f'])  # Center of regularized phase-field.
    
    r = fenics.Constant(regularization['r'])  # Regularization smoothing parameter.
    
    P = lambda T: 0.5*(1. - fenics.tanh(2.*(T_f - T)/r))  # Regularized phase field.
    
    mu_l = fenics.Constant(liquid_viscosity)
    
    mu_s = fenics.Constant(solid_viscosity)
    
    mu = lambda (T) : mu_s + (mu_l - mu_s)*P(T) # Variable viscosity.
    
    L = C/Ste  # Latent heat
    
    u_n, p_n, T_n = fenics.split(w_n)

    w_w = fenics.TrialFunction(W)
    
    u_w, p_w, T_w = fenics.split(w_w)
    
    v, q, phi = fenics.TestFunctions(W)
    
    w_k = fenics.Function(W)
    
    u_k, p_k, T_k = fenics.split(w_k)

    F = (
        b(u_k, q) - gamma*p_k*q
        + dot(u_k - u_n, v)/dt
        + c(u_k, u_k, v) + b(v, p_k) + a(mu(T_k), u_k, v)
        + dot(f_B(T_k), v)
        + C/dt*(T_k - T_n)*phi
        - dot(C*T_k*u_k, grad(phi)) 
        + K/Pr*dot(grad(T_k), grad(phi))
        + 1./dt*L*(P(T_k) - P(T_n))*phi
        )*fenics.dx

    ddT_f_B = lambda T : ddT_m_B(T)*g
    
    sech = lambda theta: 1./fenics.cosh(theta)
    
    dP = lambda T: sech(2.*(T_f - T)/r)**2/r

    dmu = lambda T : (mu_l - mu_s)*dP(T)
    
    
    # Set the Jacobian (formally the Gateaux derivative) in variational form.
    JF = (
        b(u_w, q) - gamma*p_w*q 
        + dot(u_w, v)/dt
        + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
        + a(T_w*dmu(T_k), u_k, v) + a(mu(T_k), u_w, v) 
        + dot(T_w*ddT_f_B(T_k), v)
        + C/dt*T_w*phi
        - dot(C*T_k*u_w, grad(phi))
        - dot(C*T_w*u_k, grad(phi))
        + K/Pr*dot(grad(T_w), grad(phi))
        + 1./dt*L*T_w*dP(T_k)*phi
        )*fenics.dx

        
    # Set the functional metric for the error estimator for adaptive mesh refinement.
    """I haven't found a good way to make this flexible yet.
    Ideally the user would be able to write the metric, but this would require giving the user
    access to much data that phaseflow is currently hiding.
    """
    M = P(T_k)*fenics.dx
    
    if adaptive_metric == 'phase_only':
    
        pass
        
    elif adaptive_metric == 'all':
        
        M += T_k*fenics.dx
        
        for i in range(dimensionality):
        
            M += u_k[i]*fenics.dx
            
    else:
        
        assert(False)
        
    # Make the problem.
    problem = fenics.NonlinearVariationalProblem(F, w_k, bcs, JF)
    
    
    # Make the solvers.
    """ For the purposes of this project, it would be better to just always use the adaptive solver; but
    unfortunately the adaptive solver encounters nan's whenever evaluating the error for problems not 
    involving phase-change. So far my attempts at writing a MWE to reproduce the  issue have failed.
    """   
    adaptive_solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    adaptive_solver.parameters['nonlinear_variational_solver']['newton_solver']['maximum_iterations']\
        = nlp_max_iterations
    
    adaptive_solver.parameters['nonlinear_variational_solver']['newton_solver']['absolute_tolerance']\
        = nlp_absolute_tolerance
    
    adaptive_solver.parameters['nonlinear_variational_solver']['newton_solver']['relative_tolerance']\
        = nlp_relative_tolerance

    static_solver = fenics.NonlinearVariationalSolver(problem)
    
    static_solver.parameters['newton_solver']['maximum_iterations'] = nlp_max_iterations
    
    static_solver.parameters['newton_solver']['absolute_tolerance'] = nlp_absolute_tolerance
    
    static_solver.parameters['newton_solver']['relative_tolerance'] = nlp_relative_tolerance
    
    
    # Open a context manager for the output file.
    with fenics.XDMFFile(output_dir + '/solution.xdmf') as solution_file:

    
        # Write the initial values.
        write_solution(solution_file, w_n, time) 

        if start_time >= end_time - TIME_EPS:
    
            helpers.print_once("Start time is already too close to end time. Only writing initial values.")
            
            return w_n, mesh
    
    
        # Solve each time step.
        progress = fenics.Progress('Time-stepping')

        fenics.set_log_level(fenics.PROGRESS)
        
        while time < end_time - TIME_EPS:
            
            if adaptive:
            
                adaptive_solver.solve(adaptive_solver_tolerance)
                
            else:
            
                static_solver.solve()
            
            time += time_step_size
            
            helpers.print_once("Reached time t = " + str(time))
            
            write_solution(solution_file, w_k, time)
            
            
            # Write checkpoint/restart files.
            restart_filepath = output_dir + '/restart_t' + str(time) + '.h5'
            
            with fenics.HDF5File(fenics.mpi_comm_world(), restart_filepath, 'w') as h5:
    
                h5.write(mesh.leaf_node(), 'mesh')
            
                h5.write(w_k.leaf_node(), 'w')
                
            if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
            
                with h5py.File(restart_filepath, 'r+') as h5:
                    
                    h5.create_dataset('t', data=time)
            
            
            # Check for steady state.
            if stop_when_steady and steady(W, w_k, w_n, steady_relative_tolerance):
            
                helpers.print_once("Reached steady state at time t = " + str(time))
                
                break
                
                
            # Set initial values for next time step.
            w_n.leaf_node().vector()[:] = w_k.leaf_node().vector()
            
            
            # Report progress.
            progress.update(time / end_time)
            
            if time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                helpers.print_once("Reached end time, t = " + str(end_time))
            
                break
    
    
    # Return the interpolant to sample inside of Python.
    w_k.rename('w', "state")
    
    return w_k, mesh
    
    
if __name__=='__main__':

    run()
