"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers


class State(fenics.Function):

    def __init__(self, time = 0., solution):
    
        self.time = time
        
        self.solution = solution
        
        self.spatial_dimensionality = solution.function_space().mesh.type().dim()

        
class ImplicitEulerIBVP(fenics.NonlinearVariationalProblem):

    def __init__(self, 
            state, 
            time_step_size, 
            nonlinear_variational_form, 
            boundary_conditions, 
            gateaux_derivative)
    
        fenics.NonlinearVariationalProblem.__init__(self, 
            nonlinear_variational_form, 
            state.solution, 
            boundary_conditions, 
            gateaux_derivative)

        self.state = state
    
        self.old_state = fenics.Function(state.function_space())
        
        self.old_state.leaf_node().vector()[:] = self.state.leaf_node().vector()
        

class ContinuousFunction():

    def __init__(self, function, derivative_function)
    
        self.function = function
        
        self.derivative_function = derivative_function
            

def run(
        output_dir = "phaseflow_output/",
        steady_relative_tolerance=1.e-4,
        adaptive_goal_functional = None,
        adaptive_solver_tolerance = 1.e-4,
        nlp_absolute_tolerance = 1.e-8,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        nlp_relaxation = 1.,
        quadrature_degree = None):
    """Run Phaseflow.
    
    
    Phaseflow is configured entirely through the arguments in this run() function.
    
    See the tests and examples for demonstrations of how to use this.
    """

    
    # Report arguments.
    phaseflow.helpers.print_once("Running Phaseflow with the following arguments:")
    
    phaseflow.helpers.print_once(phaseflow.helpers.arguments())
    
    phaseflow.helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + "/arguments.txt", "w")
        
        arguments_file.write(str(phaseflow.helpers.arguments()))

        arguments_file.close()
    
    
    # Use function space and mesh from initial solution.
    W = solution.function_space()
    
    mesh = W.mesh()
    
    
    # Check if 1D/2D/3D.
    dimensionality = mesh.type().dim()
    
    phaseflow.helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    
    
    
    # Make the solver.
    """ For the purposes of this project, it would be better to just always use the adaptive solver; but
    unfortunately the adaptive solver encounters nan's in some cases.
    So far my attempts at writing a MWE to reproduce the  issue have failed.
    """
    if adaptive_goal_functional is not None:
    
        adaptive = True
        
    else:
        
        adaptive = False
        
    if adaptive:
        
        solver = fenics.AdaptiveNonlinearVariationalSolver(problem = problem, 
            goal = adaptive_goal_functional)
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]["maximum_iterations"]\
            = nlp_max_iterations
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]["absolute_tolerance"]\
            = nlp_absolute_tolerance
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]["relative_tolerance"]\
            = nlp_relative_tolerance
        
        solver.parameters["nonlinear_variational_solver"]["newton_solver"]["relaxation_parameter"]\
            = nlp_relaxation
        
    else:
        
        solver = fenics.NonlinearVariationalSolver(problem)
        
        solver.parameters["newton_solver"]["maximum_iterations"] = nlp_max_iterations
        
        solver.parameters["newton_solver"]["absolute_tolerance"] = nlp_absolute_tolerance
        
        solver.parameters["newton_solver"]["relative_tolerance"] = nlp_relative_tolerance
        
        solver.parameters["newton_solver"]["relaxation_parameter"] = nlp_relaxation
    
    
    # 
    start_time = 0. + time
    
    
    # Open a context manager for the output file.
    solution_filepath = output_dir + "/solution.xdmf"
    
    with fenics.XDMFFile(solution_filepath) as solution_file:

    
        # Write the initial values.
        write_solution(solution_file, w_n, time, solution_filepath) 

        if start_time >= end_time - TIME_EPS:
    
            phaseflow.helpers.print_once("Start time is already too close to end time. Only writing initial values.")
            
            return w_n, mesh
    
    
        # Solve each time step.
        progress = fenics.Progress("Time-stepping")
        
        fenics.set_log_level(fenics.PROGRESS)
        
        for it in range(1, MAX_TIME_STEPS):
            
            if(time > end_time - TIME_EPS):
                
                break
            
            if adaptive:
            
                solver.solve(adaptive_solver_tolerance)
                
            else:
            
                solver.solve()
            
            time = start_time + it*time_step_size
            
            phaseflow.helpers.print_once("Reached time t = " + str(time))
            
            write_solution(solution_file, w, time, solution_filepath)
            
            
            # Write checkpoint files.
            write_checkpoint(checkpoint_filepath = output_dir + "/checkpoint_t" + str(time) + ".h5",
                w = w,
                time = time)
            
            
            # Check for steady state.
            if stop_when_steady and steady(W, w, w_n, steady_relative_tolerance):
            
                phaseflow.helpers.print_once("Reached steady state at time t = " + str(time))
                
                break
                
                
            # Set initial values for next time step.
            w_n.leaf_node().vector()[:] = w.leaf_node().vector()
            
            
            # Report progress.
            progress.update(time / end_time)
            
            if time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                phaseflow.helpers.print_once("Reached end time, t = " + str(end_time))
            
                break

                
if __name__=="__main__":

    run()
