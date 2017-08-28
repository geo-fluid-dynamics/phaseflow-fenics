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
import h5py
import helpers
import globals
import default
import problem
import solver
import time
import refine
import output


'''@todo First add variable viscosity, later latent heat source term.
Conceptually this will be like having a PCM with zero latent heat.
The melting front should move quickly.'''

def function_spaces(mesh=default.mesh, pressure_degree=default.pressure_degree, temperature_degree=default.temperature_degree):
    """ Define function spaces for the variational form """
    
    velocity_degree = pressure_degree + 1
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    temperature_space = fenics.FunctionSpace(mesh, 'P', temperature_degree)

    '''
    MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    '''
    @todo How can we use the space $Q = \left{q \in L^2(\Omega) | \int{q = 0}\right}$ ?

    All Navier-Stokes FEniCS examples I've found simply use P2P1. danaila2014newton says that
    they are using the "classical Hilbert spaces" for velocity and pressure, but then they write
    down the space Q with less restrictions than H^1_0.

    '''
    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    solution_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])

    solution_function_space = fenics.FunctionSpace(mesh, solution_element)  
    
    return solution_function_space, solution_element


def run(
    output_dir = 'output/natural_convection',
    Ra = default.parameters['Ra'],
    Pr = default.parameters['Pr'],
    Ste = default.parameters['Ste'],
    C = default.parameters['C'],
    K = default.parameters['K'],
    mu_l = default.parameters['mu_l'],
    mu_s = default.parameters['mu_s'],
    g = default.parameters['g'],
    m_B = default.m_B,
    ddtheta_m_B = default.ddtheta_m_B,
    regularization = default.regularization,
    mesh=default.mesh,
    initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
    boundary_conditions = [{'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': 'topological'}],
    start_time = 0.,
    end_time = 1.,
    time_step_bounds = (1.e-3, 1.e-3, 1.),
    output_times = ('start', 'end'),
    max_pci_refinement_cycles = 0,
    initial_pci_refinement_cycles = 0,
    gamma = 1.e-7,
    custom_newton = True,
    nlp_absolute_tolerance = 1.,
    nlp_relative_tolerance = 1.e-8,
    nlp_max_iterations = 12,
    nlp_divergence_threshold = 1.e12,
    nlp_relaxation = 1.,
    pressure_degree = default.pressure_degree,
    temperature_degree = default.temperature_degree,
    automatic_jacobian = True,
    stop_when_steady = False,
    steady_relative_tolerance = 1.e-8,
    restart = False,
    restart_filepath = '',
    debug = False):
    
        
    # Display inputs
    helpers.print_once("Running Phaseflow with the following arguments:")
    
    helpers.print_once(helpers.arguments())
    
    helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + '/arguments.txt', 'w')
        
        arguments_file.write(str(helpers.arguments()))

        arguments_file.close()
    
    
    # Validate inputs    
    if type(time_step_bounds) == type(1.):
    
        time_step_bounds = (time_step_bounds, time_step_bounds, time_step_bounds)
    
    time_step_size = time.TimeStepSize(helpers.BoundedValue(time_step_bounds[0], time_step_bounds[1], time_step_bounds[2]))
        
    dimensionality = mesh.type().dim()
    
    helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Initialize time
    if restart:
    
        with h5py.File(restart_filepath, 'r') as h5:
            
            current_time = h5['t'].value
            
            assert(abs(current_time - start_time) < time.TIME_EPS)
    
    else:
    
        current_time = start_time
    
    
    # Open the output file(s)   
    ''' @todo  explore info(f.parameters, verbose=True) 
    to avoid duplicate mesh storage when appropriate 
    per https://fenicsproject.org/qa/3051/parallel-output-of-a-time-series-in-hdf5-format '''

    with fenics.XDMFFile(output_dir + '/solution.xdmf') as solution_file:

        # Solve each time step
        progress = fenics.Progress('Time-stepping')

        fenics.set_log_level(fenics.PROGRESS)
            
        output_count = 0
        
        if (output_times is not ()):
        
            if output_times[0] == 'start':
            
                output_start_time = True
                
                output_count += 1
            
            if output_times[0] == 'all':
            
                output_start_time = True
            
        else:
        
            output_start_time = False
            
        if stop_when_steady:
        
            steady = False
        
        pci_refinement_cycle = 0
        
        while current_time < (end_time - fenics.dolfin.DOLFIN_EPS):
        
            time_step_size, next_time, output_this_time, output_count = time.check(current_time,
                time_step_size, end_time, output_times, output_count)
            
            for ir in range(max_pci_refinement_cycles + 1):
            
            
                # Define function spaces and solution function 
                W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)

                w = fenics.Function(W)
                
                
                # Set the initial values
                if restart:
                
                    mesh = fenics.Mesh()
                    
                    with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
                    
                        h5.read(mesh, 'mesh', True)
                        
                    W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
                
                    w_n = fenics.Function(W)
                    
                    with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
                    
                        h5.read(w_n, 'w')
                        
                    w = fenics.Function(W)
                    
                else:
                    
                    if fenics.near(current_time, start_time):
            
                        w_n = fenics.interpolate(fenics.Expression(initial_values_expression,
                            element=W_ele), W)
                    
                    else:
            
                        w_n = fenics.project(w_n, W)

                if pci_refinement_cycle < initial_pci_refinement_cycles:
                
                    mesh = refine.refine_pci(regularization, pci_refinement_cycle, mesh, w_n)
                    
                    pci_refinement_cycle += 1
                    
                    continue
                    
                    
                # Initialize the functions that we will use to generate our variational form
                form_factory = problem.FormFactory(W, {'Ra': Ra, 'Pr': Pr, 'Ste': Ste, 'C': C, 'K': K, 'g': g, 'gamma': gamma, 'mu_l': mu_l, 'mu_s': mu_s}, m_B, ddtheta_m_B, regularization)

                
                # Make the time step solver
                solve_time_step = solver.make(form_factory = form_factory,
                    nlp_absolute_tolerance = nlp_absolute_tolerance,
                    nlp_relative_tolerance = nlp_relative_tolerance,
                    nlp_max_iterations = nlp_max_iterations,
                    nlp_divergence_threshold = nlp_divergence_threshold,
                    nlp_relaxation = nlp_relaxation,
                    custom_newton = custom_newton,
                    automatic_jacobian = automatic_jacobian)
                
                
                # Organize boundary conditions
                bcs = []
                
                for item in boundary_conditions:
                
                    bcs.append(fenics.DirichletBC(W.sub(item['subspace']), item['value_expression'],
                        item['location_expression'], method=item['method']))
                

                # Write the initial values                    
                if output_start_time and fenics.near(current_time, start_time):
                    
                    output.write_solution(solution_file, w_n, current_time) 
                 
                 
                #
                converged = time.adaptive_time_step(time_step_size=time_step_size, w=w, w_n=w_n, bcs=bcs,
                    solve_time_step=solve_time_step, debug=debug)
                
                
                #
                if converged:
                
                    break
                
                
                # Refine mesh cells containing the PCI
                if (max_pci_refinement_cycles is 0) or (pci_refinement_cycle
                        is (max_pci_refinement_cycles - 1)):

                    break
                    
                mesh = refine.refine_pci(regularization, pci_refinement_cycle, mesh, w) # @todo Use w_n or w?
                    
                pci_refinement_cycle = pci_refinement_cycle + 1
                
            assert(converged)
            
            current_time += time_step_size.value
            
            if stop_when_steady and time.steady(W, w, w_n):
            
                steady = True
                
                if output_times[-1] == 'end':
                
                    output_this_time = True
            
            if output_this_time:
            
                output.write_solution(solution_file, w, current_time)
                
                # Write checkpoint/restart files
                restart_filepath = output_dir+'/restart_t'+str(current_time)+'.hdf5'
                
                with fenics.HDF5File(fenics.mpi_comm_world(), restart_filepath, 'w') as h5:
        
                    h5.write(mesh, 'mesh')
                
                    h5.write(w, 'w')
                    
                if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
                
                    with h5py.File(restart_filepath, 'r+') as h5:
                        
                        h5.create_dataset('t', data=current_time)
                        
            helpers.print_once("Reached time t = " + str(current_time))
                
            if stop_when_steady and steady:
            
                helpers.print_once("Reached steady state at time t = " + str(current_time))
                
                break

            w_n.assign(w)  # The current solution becomes the new initial values
            
            progress.update(current_time / end_time)
            
            time_step_size.set(2*time_step_size.value) # @todo: Encapsulate the adaptive time stepping
                
    if current_time >= (end_time - fenics.dolfin.DOLFIN_EPS):
    
        helpers.print_once("Reached end time, t = "+str(end_time))
    
    
    # Return the interpolant to sample inside of Python
    w_n.rename('w', "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
    
    return fe_field_interpolant, mesh
    
    
if __name__=='__main__':

    run()
