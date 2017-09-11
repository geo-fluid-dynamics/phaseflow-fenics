"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import helpers
import globals
import default
import form
import solver
import bounded_value
import time
import refine
import output


'''@todo First add variable viscosity, later latent heat source term.
Conceptually this will be like having a PCM with zero latent heat.
The melting front should move quickly.'''

def function_spaces(mesh=default.mesh, pressure_degree=default.pressure_degree, temperature_degree=default.temperature_degree):
    """ Define function spaces for the variational form."""
    
    velocity_degree = pressure_degree + 1
    
    velocity_space = fenics.VectorFunctionSpace(mesh, 'P', velocity_degree)

    pressure_space = fenics.FunctionSpace(mesh, 'P', pressure_degree) # @todo mixing up test function space

    temperature_space = fenics.FunctionSpace(mesh, 'P', temperature_degree)

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    '''
    @todo How can we use the space
        $Q = \left{q \in L^2(\Omega) | \int{q = 0}\right}$ ?

    All Navier-Stokes FEniCS examples I've found simply use P2P1. danaila2014newton says that
    they are using the "classical Hilbert spaces" for velocity and pressure, but then they write down the space Q with less restrictions than H^1_0.
    
    My understanding is that the space Q relates to the divergence-free
    requirement.
    '''
    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    solution_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])

    solution_function_space = fenics.FunctionSpace(mesh, solution_element)  
    
    return solution_function_space, solution_element


def run(
    output_dir = 'output/steady_lid_driven_cavity_Re1',
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
    initial_values_expression = (lid, "0.", "0.", "1."),
    boundary_conditions = [
        {'subspace': 0, 'value_expression': ("1.", "0."), 'degree': 3, 'location_expression': 'near(x[1],  1.)',
        'method': 'topological'},
        {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)',
        'method': 'topological'},
        {'subspace': 1, 'value_expression': "0.", 'degree': 2, 'location_expression': 'near(x[0], 0.) && near(x[1], 0.)',
        'method': 'pointwise'},
        {'subspace': 2, 'value_expression': "1.", 'degree': 2, 'location_expression': 'near(x[0], 0.) && near(x[1], 0.)',
        'method': 'pointwise'}],
    start_time = 0.,
    end_time = 1.e12,
    time_step_bounds = (1.e12, 1.e12, 1.e12),
    output_times = ('all',),
    max_time_steps = 1000,
    max_pci_refinement_cycles = 1000,
    max_pci_refinement_cycles_per_time = 0,
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
    automatic_jacobian = False,
    stop_when_steady = False,
    steady_relative_tolerance = 1.e-4,
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
    
    time_step_size = time.TimeStepSize(
        bounded_value.BoundedValue(time_step_bounds[0],
            time_step_bounds[1], time_step_bounds[2]))
        
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
                
            if output_times[0] == 'end':
            
                output_start_time = False
            
        else:
        
            output_start_time = False
            
        if stop_when_steady:
        
            steady = False
        
        pci_refinement_cycle = 0
        
        for it in range(max_time_steps):
        
            pci_refinement_cycle_this_time = 0
            
            while (pci_refinement_cycle < (max_pci_refinement_cycles + 1)) and (
                pci_refinement_cycle_this_time < (max_pci_refinement_cycles_per_time + 1)):
            
            
                # Define function spaces and solution function 
                W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)

                w = fenics.Function(W)
                
                
                # Set the initial values
                if (abs(current_time - start_time) < time.TIME_EPS):
                
                    if restart:
                
                        if pci_refinement_cycle == 0:
                        
                            mesh = fenics.Mesh()
                            
                            with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
                            
                                h5.read(mesh, 'mesh', True)
                            
                            W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)
                        
                            w_n = fenics.Function(W)
                        
                            with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, 'r') as h5:
                            
                                h5.read(w_n, 'w')
                            
                            w = fenics.Function(W)
                            
                        else:
                        
                            w_n = fenics.interpolate(w_n, W)
                    
                    else:
            
                        w_n = fenics.interpolate(fenics.Expression(initial_values_expression,
                            element=W_ele), W)
                    
                else:
            
                    w_n = fenics.project(w_n, W)

                if pci_refinement_cycle < initial_pci_refinement_cycles:
                
                    mesh = refine.refine_pci(regularization, pci_refinement_cycle, mesh, w_n)
                    
                    pci_refinement_cycle += 1
                    
                    continue
                    
                if start_time >= end_time - time.TIME_EPS:
                
                    helpers.print_once("Start time is already too close to end time. Only writing initial values.")
                    
                    output.write_solution(solution_file, w_n, current_time)
                    
                    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
                    
                    return fe_field_interpolant, mesh
                
                time_step_size, next_time, output_this_time, output_count, next_output_time = time.check(current_time,
                time_step_size, end_time, output_times, output_count)
                
                # Initialize the functions that we will use to generate our variational form
                form_factory = form.FormFactory(W, {'Ra': Ra, 'Pr': Pr, 'Ste': Ste, 'C': C, 'K': K, 'g': g, 'gamma': gamma, 'mu_l': mu_l, 'mu_s': mu_s}, m_B, ddtheta_m_B, regularization)

                
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
                if (max_pci_refinement_cycles_per_time == 0) or (pci_refinement_cycle_this_time
                        == max_pci_refinement_cycles_per_time):

                    break
                    
                mesh = refine.refine_pci(regularization, pci_refinement_cycle_this_time, mesh, w) # @todo Use w_n or w?
                
                pci_refinement_cycle += 1
                
                pci_refinement_cycle_this_time += 1
                
            assert(converged)
            
            current_time += time_step_size.value
            
            if stop_when_steady and time.steady(W, w, w_n, 
                    steady_relative_tolerance):
            
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
            
            if current_time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                helpers.print_once("Reached end time, t = "+str(end_time))
            
                break
            
            time_step_size.set(2*time_step_size.value) # @todo: Encapsulate the adaptive time stepping
                
    # Return the interpolant to sample inside of Python
    w_n.rename('w', "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
    
    return fe_field_interpolant, mesh
    
    
if __name__=='__main__':

    run()
