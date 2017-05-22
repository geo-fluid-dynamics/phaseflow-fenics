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
import globals
import default
import forms
import solver
import time
import output


'''@todo First add variable viscosity, later latent heat source term.
Conceptually this will be like having a PCM with zero latent heat.
The melting front should move quickly.'''

def function_spaces(mesh=default.mesh, pressure_degree=default.pressure_degree, temperature_degree=default.temperature_degree):
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
    output_dir = 'output/natural_convection',
    Ra = default.parameters['Ra'],
    Pr = default.parameters['Pr'],
    K = default.parameters['K'],
    mu_l = default.parameters['mu_l'],
    g = default.parameters['g'],
    m_B = default.m_B,
    ddtheta_m_B = default.ddtheta_m_B,
    mesh=default.mesh,
    initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
    boundary_conditions = [{'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': 'topological'}, {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': 'topological'}],
    final_time = 1.,
    time_step_bounds = (1.e-3, 1.e-3, 1.),
    adaptive_space = False,
    adaptive_space_error_tolerance = 1.e-4,
    gamma = 1.e-7,
    pressure_degree = default.pressure_degree,
    temperature_degree = default.temperature_degree,
    linearize = True,
    stop_when_steady = False,
    steady_relative_tolerance = 1.e-8):

    # Display inputs
    print("Running Phaseflow with the following arguments:")
    
    print(helpers.arguments())
    
    
    # Validate inputs    
    if type(time_step_bounds) == type(1.):
    
        time_step_bounds = (time_step_bounds, time_step_bounds, time_step_bounds)
    
    time_step_size = time.TimeStepSize(helpers.BoundedValue(time_step_bounds[0], time_step_bounds[1], time_step_bounds[2]))
        
        
    # Define function spaces and solution function
    W, W_ele = function_spaces(mesh, pressure_degree, temperature_degree)

    w = fenics.Function(W)
    
    
    # Set the initial values
    w_n = fenics.interpolate(fenics.Expression(initial_values_expression, element=W_ele), W)

    
    # Initialize the functions that we will use to generate our variational form
    form_factory = forms.FormFactory(W, {'Ra': Ra, 'Pr': Pr, 'K': K, 'g': g, 'gamma': gamma, 'mu_l': mu_l}, m_B, ddtheta_m_B)

    
    # Organize boundary conditions
    bcs = []
    
    for item in boundary_conditions:
    
        bcs.append(fenics.DirichletBC(W.sub(item['subspace']), fenics.Expression(item['value_expression'], degree=item['degree']), item['location_expression'], method=item['method']))
    
    
    # Open the output VTK files, and write initial values
    solution_files = [fenics.File(output_dir + '/velocity.pvd'), fenics.File(output_dir + '/pressure.pvd'), fenics.File(output_dir + '/temperature.pvd')]

    current_time = 0.
    
    output.write_solution(solution_files, w_n, current_time) 

    
    # Solve each time step
    solve_time_step = solver.make(form_factory, linearize, adaptive_space, adaptive_space_error_tolerance)
    
    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
    
    while current_time < final_time - dolfin.DOLFIN_EPS:

        remaining_time = final_time - current_time
    
        if time_step_size.value > remaining_time:
            
            time_step_size.set(remaining_time)

        current_time, converged = time.adaptive_time_step(time_step_size=time_step_size, w=w, w_n=w_n, bcs=bcs, current_time=current_time, solve_time_step=solve_time_step)
    
        ''' Save solution to files.
        Saving here allows us to inspect the latest solution 
        even if the Newton iterations failed to converge.'''
        output.write_solution(solution_files, w, current_time)
        
        assert(converged)
        
        time_step_size.set(2*time_step_size.value) # @todo: Encapsulate the adaptive time stepping
                    
        print 'Reached time t = ' + str(current_time)
        
        if stop_when_steady and time.steady(W, w, w_n):
        
            print 'Reached steady state at time t = ' + str(current_time)
            
            break

        w_n.assign(w)
        
        progress.update(current_time / final_time)
        
    
    if time >= (final_time - dolfin.DOLFIN_EPS):
    
        print 'Reached final time, t = ' + str(final_time)
        
    w_n.rename("w", "state")
        
    fe_field_interpolant = fenics.interpolate(w_n.leaf_node(), W)
        
    return fe_field_interpolant
    
    
if __name__=='__main__':

    run()