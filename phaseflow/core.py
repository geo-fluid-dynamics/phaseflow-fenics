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
import forms
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
    output_format = 'vtk',
    Ra = default.parameters['Ra'],
    Pr = default.parameters['Pr'],
    Ste = default.parameters['Ste'],
    C = default.parameters['C'],
    K = default.parameters['K'],
    mu_l = default.parameters['mu_l'],
    g = default.parameters['g'],
    m_B = default.m_B,
    ddtheta_m_B = default.ddtheta_m_B,
    regularization = default.regularization,
    mesh=default.mesh,
    initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
    boundary_conditions = [{'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': 'topological'}],
    final_time = 1.,
    time_step_bounds = (1.e-3, 1.e-3, 1.),
    output_times = ('initial', 1.e-3, 1.e-2, 1.e-1, 'final'),
    max_pci_refinement_cycles = 0,
    adaptive_space = False,
    adaptive_space_error_tolerance = 1.e-4,
    gamma = 1.e-7,
    newton_relative_tolerance = 1.e-8,
    max_newton_iterations = 12,
    pressure_degree = default.pressure_degree,
    temperature_degree = default.temperature_degree,
    linearize = True,
    stop_when_steady = False,
    steady_relative_tolerance = 1.e-8,
    restart = False,
    debug = False):
    
        
    # Display inputs
    helpers.print_once("Running Phaseflow with the following arguments:")
    
    helpers.print_once(helpers.arguments())
    
    helpers.mkdir_p(output_dir)
    
    if fenics.dolfin.MPI.rank(fenics.dolfin.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + '/arguments.txt', 'w')
        
        arguments_file.write(str(helpers.arguments()))

        arguments_file.close()
    
    
    # Validate inputs    
    if type(time_step_bounds) == type(1.):
    
        time_step_bounds = (time_step_bounds, time_step_bounds, time_step_bounds)
    
    time_step_size = time.TimeStepSize(helpers.BoundedValue(time_step_bounds[0], time_step_bounds[1], time_step_bounds[2]))
        
    dimensionality = mesh.type().dim()
    
    helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Set derived parameters.
    restart_filename = output_dir+"/restart.hdf5"
   
    
    # Initialize time
    if restart:
    
        with h5py.File(restart_filename, "r") as h5:
                    
            current_time = h5['t'].value
    
    else:
    
        current_time = 0.    
    
    
    # Open the output file(s)   
    if output_format is 'vtk':
    
        solution_files = [fenics.File(output_dir + '/velocity.pvd'), fenics.File(output_dir + '/pressure.pvd'), fenics.File(output_dir + '/temperature.pvd')]
        
    elif output_format is 'xdmf': # Parallel format
    
        ''' @todo  explore info(f.parameters, verbose=True) 
        to avoid duplicate mesh storage when appropriate 
        per https://fenicsproject.org/qa/3051/parallel-output-of-a-time-series-in-hdf5-format '''
    
        solution_files = [fenics.XDMFFile(output_dir + '/velocity.xdmf'), fenics.XDMFFile(output_dir + '/pressure.xdmf'), fenics.XDMFFile(output_dir + '/temperature.xdmf')]

    elif output_format is 'table':        
    
        solution_files = [open(output_dir + 'temperature.txt', 'w')]

    
    # Solve each time step
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
        
    output_count = 0
    
    if (output_times is not ()) and (output_times[0] == 'initial'):
    
        output_initial_time = True
        
        output_count += 1
        
    else:
    
        output_initial_time = False
        
    if stop_when_steady:
    
        steady = False
    
    while current_time < (final_time - fenics.dolfin.DOLFIN_EPS):
    
        time_step_size, next_time, output_this_time, output_count = time.check(current_time,
            time_step_size, final_time, output_times, output_count)
        
        pci_refinement_cycle = 0
            
        for ir in range(max_pci_refinement_cycles + 1):
        
            # Define function spaces and solution function
            W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)

            w = fenics.Function(W)
            
            
            # Set the initial values
            if fenics.near(current_time, 0.):
            
                w_n = fenics.interpolate(fenics.Expression(initial_values_expression,
                    element=W_ele), W)
                    
            else:
            
                if restart and (ir == 0):
                
                    w_n = fenics.Function(W)
                
                    mesh = fenics.Mesh()
        
                    with fenics.HDF5File(mesh.mpi_comm(), restart_filename, 'r') as h5:
                    
                        h5.read(mesh, "mesh", True)
                    
                        h5.read(w_n, "w_n")
                    
                else:
            
                    w_n = fenics.project(w_n, W)

            
            # Initialize the functions that we will use to generate our variational form
            form_factory = forms.FormFactory(W, {'Ra': Ra, 'Pr': Pr, 'Ste': Ste, 'C': C, 'K': K, 'g': g, 'gamma': gamma, 'mu_l': mu_l}, m_B, ddtheta_m_B, regularization)

            
            # Make the time step solver
            solve_time_step = solver.make(form_factory, newton_relative_tolerance, max_newton_iterations, linearize, adaptive_space, adaptive_space_error_tolerance)
            
            
            # Organize boundary conditions
            bcs = []
            
            for item in boundary_conditions:
            
                bcs.append(fenics.DirichletBC(W.sub(item['subspace']), item['value_expression'], item['location_expression'], method=item['method']))
            

            # Write the initial values                    
            if output_initial_time and fenics.near(current_time, 0.) and (ir is 0):
            
                if output_format is 'table':
                
                    solution_files[0].write("t, x, theta \n")
                
                output.write_solution(output_format, solution_files, W, w_n, current_time) 
             
             
            #
            converged = time.adaptive_time_step(time_step_size=time_step_size, w=w, w_n=w_n, bcs=bcs,
                solve_time_step=solve_time_step)
            
            
            #
            if converged:
            
                break
            
            
            # Refine mesh cells containing the PCI
            if (max_pci_refinement_cycles is 0) or (pci_refinement_cycle
                    is (max_pci_refinement_cycles - 1)):

                break
                
            pci_refinement_cycle = pci_refinement_cycle + 1
            
            contains_pci = refine.mark_pci_cells(regularization, mesh, w)

            pci_cell_count = sum(contains_pci)

            assert(pci_cell_count > 0)

            helpers.print_once("PCI Refinement cycle #"+str(pci_refinement_cycle)+
                ": Refining "+str(pci_cell_count)+" cells containing the PCI.")

            mesh = fenics.refine(mesh, contains_pci)                    
                
        assert(converged)
        
        current_time += time_step_size.value
        
        if stop_when_steady and time.steady(W, w, w_n):
        
            steady = True
            
            if output_times[output_count] == 'final':
            
                output_this_time = True
        
        if output_this_time:
        
            output.write_solution(output_format, solution_files, W, w, current_time)
            
            # Write checkpoint/restart files
            with fenics.HDF5File(fenics.mpi_comm_world(), restart_filename, "w") as h5:
    
                h5.write(mesh, "mesh")
            
                h5.write(w_n, "w_n")
                
            with h5py.File(restart_filename, "r+") as h5:
                
                h5.create_dataset("t", data=current_time)
                    
        helpers.print_once('Reached time t = ' + str(current_time))
            
        if stop_when_steady and steady:
        
            helpers.print_once('Reached steady state at time t = ' + str(current_time))
            
            break

        w_n.assign(w)  # The current solution becomes the new initial values
        
        progress.update(current_time / final_time)
        
        time_step_size.set(2*time_step_size.value) # @todo: Encapsulate the adaptive time stepping
            
    if current_time >= (final_time - fenics.dolfin.DOLFIN_EPS):
    
        helpers.print_once("Reached final time, t = "+str(final_time))
    
    
    # Clean up
    if output_format is 'table':
    
        solution_files[0].close()
    
    
    # Return the interpolant to sample inside of Python
    w_n.rename("w", "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
    
    return fe_field_interpolant, mesh
    
    
if __name__=='__main__':

    run()