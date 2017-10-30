"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import helpers
import globals
import default
import form
import solver
import bounded_value
import refine
import output


'''@todo First add variable viscosity, later latent heat source term.
Conceptually this will be like having a PCM with zero latent heat.
The melting front should move quickly.'''

def make_mixed_fe(cell, pressure_degree=default.pressure_degree, temperature_degree=default.temperature_degree):
    """ Define the mixed finite element.
    MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    """
    velocity_degree = pressure_degree + 1
    
    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    solution_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])
    
    return solution_element


def steady(W, w, w_n, steady_relative_tolerance=1.e-4):
    '''Check if solution has reached an approximately steady state.'''
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
    
    helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady
    
    
def run(
    output_dir = 'output/wang2010_natural_convection_air',
    rayleight_number = default.parameters['Ra'],
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
    stop_when_steady = False,
    adaptive_solver_tolerance = 1.e-4,
    nlp_relative_tolerance = 1.e-4,
    nlp_max_iterations = 12,
    pressure_degree = default.pressure_degree,
    temperature_degree = default.temperature_degree,
    restart = False,
    restart_filepath = '',
    debug = False):
    """Run Phaseflow.
    
    Rather than using an input file, Phaseflow is configured entirely through
    the arguments in this run() function.
    
    See the tests and examples for demonstrations of how to use this.
    """
    
    '''@todo Describe the arguments in the docstring.
    Phaseflow has been in rapid development and these have been changing.
    Now that things are stabilizing somewhat, it's about time to document
    these arguments properly.
    '''
    
    # Display inputs.
    helpers.print_once("Running Phaseflow with the following arguments:")
    
    helpers.print_once(helpers.arguments())
    
    helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + '/arguments.txt', 'w')
        
        arguments_file.write(str(helpers.arguments()))

        arguments_file.close()
    
    
    #
    dimensionality = mesh.type().dim()
    
    helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Initialize time.
    if restart:
    
        with h5py.File(restart_filepath, 'r') as h5:
            
            current_time = h5['t'].value
            
            assert(abs(current_time - start_time) < time.TIME_EPS)
    
    else:
    
        current_time = start_time
    
    
    # Define function spaces and solution function.
    W_ele = function_spaces(mesh.ufl_cell(), pressure_degree, temperature_degree)
    
    W = fenics.FunctionSpace(mesh, W_ele)
    
    
    # Set the initial values.
    if restart:
            
        mesh = fenics.Mesh()
        
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
        
            h5.read(mesh, 'mesh', True)
        
        W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
    
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
    
    a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)  # Convection of the velocity field
    
    dt = fenics.Constant(time_step_size)
    
    Ra = fenics.Constant(rayleight_number), 
    
    Pr = fenics.Constant(prandtl_number)
    
    Ste = fenics.Constant(stefan_number)
    
    C = fenics.Constant(heat_capacity)
    
    K = fenics.Constant(thermal_conductivity)

    g = fenics.Constant(gravity)
    
    gamma = fenics.Constant(penalty_parameter)
    
    mu_l = fenics.Constant(liquid_viscosity)
    
    mu_s = fenics.Constant(solid_viscosity)
    
    f_B = lambda T : m_B(T)*g  # Buoyancy force, $f = ma$
    
    T_f = fenics.Constant(regularization['T_f'])  # Center of regularized phase-field.
    
    r = fenics.Constant(regularization['r'])  # Regularization smoothing parameter.
    
    L = C/Ste  # Latent heat
    
    P = lambda T: 0.5*(1. - fenics.tanh(2.*(T_f - T)/r))  # Regularized phase field.
    
    mu = lambda (T) : mu_s + (mu_l - mu_s)*P(T) # Variable viscosity.
    
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
    M = P(T_k)*fenics.dx
    
    
    # Make the problem.
    problem = fenics.NonlinearVariationalProblem(F, w_k, bcs, JF)
    
    
    # Make the solver.
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    solver.parameters['nonlinear_variational_solver']['newton_solver']['maximum_iterations'] = nlp_max_iterations
    
    solver.parameters['nonlinear_variational_solver']['newton_solver']['relative_tolerance'] = nlp_relative_tolerance

    solver.parameters['nonlinear_variational_solver']['newton_solver']['error_on_nonconvergence'] = True

    ''' @todo  explore info(f.parameters, verbose=True) 
    to avoid duplicate mesh storage when appropriate 
    per https://fenicsproject.org/qa/3051/parallel-output-of-a-time-series-in-hdf5-format '''

    with fenics.XDMFFile(output_dir + '/solution.xdmf') as solution_file:

        # Write the initial values.
        output.write_solution(solution_file, w_n, current_time) 

        if start_time >= end_time - time.TIME_EPS:
    
            helpers.print_once("Start time is already too close to end time. Only writing initial values.")
            
            fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
            
            return fe_field_interpolant, mesh
    
    
        # Solve each time step.
        progress = fenics.Progress('Time-stepping')

        fenics.set_log_level(fenics.PROGRESS)
        
        while time < end_time - time.TIME_EPS:
            
            solver.solve(adaptive_solver_tolerance)
            
            time += time_step_size
            
            output.write_solution(solution_file, w_k, time)
            
            
            # Write checkpoint/restart files.
            restart_filepath = output_dir + '/restart_t' + str(time) + '.h5'
            
            with fenics.HDF5File(fenics.mpi_comm_world(), restart_filepath, 'w') as h5:
    
                h5.write(mesh.leaf_node(), 'mesh')
            
                h5.write(w_k.leaf_node(), 'w')
                
            if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
            
                with h5py.File(restart_filepath, 'r+') as h5:
                    
                    h5.create_dataset('t', data=current_time)
                        
            helpers.print_once("Reached time t = " + str(current_time))
            
            
            # Check for steady state.
            if stop_when_steady and time.steady(W, w_k, w_n, steady_relative_tolerance):
            
                helpers.print_once("Reached steady state at time t = " + str(current_time))
                
                break
                
                
            # Set initial values for next time step.
            w_n.leaf_node().vector()[:] = w_k.leaf_node().vector()
            
            
            # Report progress.
            progress.update(current_time / end_time)
            
            if current_time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                helpers.print_once("Reached end time, t = " + str(end_time))
            
                break
    
    
    # Return the interpolant to sample inside of Python.
    w_n.rename('w', "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
    
    return fe_field_interpolant, mesh
    
    
if __name__=='__main__':

    run()
