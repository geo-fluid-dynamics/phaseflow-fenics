"""**simulation.py** contains the Simulation class.

This is a 'God class' which collects all of Phaseflow's primary data and methods.
"""
import fenics
import numpy
import phaseflow.helpers
import tempfile
import pickle


class Simulation:

    def __init__(self):
    
        self.mesh = None
        
        self.element = None
        
        self.function_space = None
        
        self.state = None
        
        self.old_state = None
        
        self.governing_form = None
        
        self.boundary_conditions = None
        
        self.problem = None
        
        self.adaptive_goal_form = None
        
        self.solver = None
        
        self.solution_file = None
        
        self.end_time = None
        
        self.unsteadiness = None
        
        self.quadrature_degree = None
        
        self.integration_metric = fenics.dx
        
        self.timestep_size = 1.
        
        self.minimum_timestep_size = 1.e-4
        
        self.maximum_timestep_size = 1.e12
        
        self.adaptive_goal_tolerance = 1.e12

        self.nonlinear_solver_max_iterations = 50
        
        self.nonlinear_solver_absolute_tolerance = 1.e-10
        
        self.nonlinear_solver_relative_tolerance = 1.e-9
        
        self.nonlinear_solver_relaxation = 1.
        
        self.prefix_output_dir_with_tempdir = False
        
        self.output_dir = "phaseflow/output/"
        
        self.stop_when_steady = False
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = False
        
        self.adaptive_time_power = 1.
        
        self.time_epsilon = 1.e-8
    
        self.max_timesteps = 1000000000000
        
        
    def update(self):
    
        self.update_element()
    
        self.function_space = fenics.FunctionSpace(self.mesh, self.element)
        
        self.state = phaseflow.state.State(self.function_space, self.element)

        self.old_state = phaseflow.state.State(self.function_space, self.element)
        
        self.update_initial_values()
        
        self.update_governing_form()
        
        self.update_problem()
        
        if self.quadrature_degree is not None:
        
            self.integration_metric = fenics.dx(metadata={'quadrature_degree': self.quadrature_degree})
        
        self.update_adaptive_goal_form()
        
        self.update_solver()
        
        self.update_initial_guess()
    
    
    def update_element(self):
        """ This must be overloaded. """
        assert(self.element is not None)
    
    
    def update_initial_values(self):
        """ This must be overloaded. """
        assert(False)
        
        
    def update_governing_form(self):
        """ Set the variational form for the governing equations.
        
        This must be overloaded.
        
        Optionally, self.derivative_of_governing_form can be set here.
        Otherwise, the derivative will be computed automatically.
        """
        assert(type(self.governing_form) is type(fenics.NonlinearVariationalForm))
        
        
    def update_problem(self):

        derivative_of_governing_form = fenics.derivative(self.governing_form, 
            self.state.solution, 
            fenics.TrialFunction(self.function_space))
        
        fenics_bcs = []
        
        for dict in self.boundary_conditions:
        
            fenics_bcs.append(
                fenics.DirichletBC(self.function_space.sub(dict["subspace"]), 
                    dict["value"], 
                    dict["location"]))
                    
        self.problem = fenics.NonlinearVariationalProblem( 
            self.governing_form, 
            self.state.solution, 
            fenics_bcs, 
            derivative_of_governing_form)
        
    
    def update_adaptive_goal_form(self):
        """ Overload this to set the goal for adaptive mesh refinement. """
        self.adaptive_goal_form = self.state.solution[0]*self.integration_metric
        
        
    def update_solver(self):
    
        self.solver = fenics.AdaptiveNonlinearVariationalSolver(
            problem = self.problem,
            goal = self.adaptive_goal_form)

        self.solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["maximum_iterations"] = self.nonlinear_solver_max_iterations
    
        self.solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["absolute_tolerance"] = self.nonlinear_solver_absolute_tolerance
        
        self.solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["relative_tolerance"] = self.nonlinear_solver_relative_tolerance
        
        self.solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
            ["relaxation_parameter"] = self.nonlinear_solver_relaxation

    
    def update_initial_guess(self):
        """ Overload this to set a different initial guess for the Newton solver. """
        self.state.solution.leaf_node().vector()[:] = self.old_state.solution.leaf_node().vector()

        
    def run(self):
        
        self.update()
        
        if self.prefix_output_dir_with_tempdir:
        
            self.output_dir = tempfile.mkdtemp() + "/" + self.output_dir
        
        solution_filepath = self.output_dir + "/solution.xdmf"
    
        with phaseflow.helpers.SolutionFile(solution_filepath) as self.solution_file:
            """ Run inside of a file context manager.
            Without this, exceptions are more likely to corrupt the outputs.
            """
            self.old_state.write_solution(self.solution_file)
        
            start_time = 0. + self.state.time

            if self.end_time is not None:
            
                if start_time >= self.end_time - self.time_epsilon:
            
                    phaseflow.helpers.print_once(
                        "Start time is already too close to end time. Only writing initial values.")
                    
                    return
                
                
            # Run solver for each time step until reaching end time.
            progress = fenics.Progress("Time-stepping")
            
            fenics.set_log_level(fenics.PROGRESS)
            
            for it in range(1, self.max_timesteps):
                
                if self.end_time is not None:
                
                    if(self.state.time > self.end_time - self.time_epsilon):
                        
                        break
                
                self.solver.solve(self.adaptive_goal_tolerance)
            
                self.state.time += self.timestep_size

                phaseflow.helpers.print_once("Reached time t = " + str(self.state.time))
                
                self.state.write_solution(self.solution_file)

                self.write_checkpoint()
                
                
                # Check for steady state.
                if self.stop_when_steady:
                
                    self.set_unsteadiness()
                    
                    if (self.unsteadiness < self.steady_relative_tolerance):
                
                        steady = True
                        
                        phaseflow.helpers.print_once("Reached steady state at time t = " 
                            + str(self.state.time))
                        
                        break
                        
                    if self.adapt_timestep_to_unsteadiness:

                        new_timestep_size = self.timestep_size/self.unsteadiness**self.adaptive_time_power
                        
                        if new_timestep_size > self.maximum_timestep_size:
                        
                            new_timestep_size = self.maximum_timestep_size
                            
                        if new_timestep_size < self.minimum_timestep_size:
                        
                            new_timestep_size = self.minimum_timestep_size
                        
                        self.timestep_size = 0. + new_timestep_size
                        
                            
                if self.end_time is not None:
                
                    progress.update(self.state.time / self.end_time)
                    
                    if self.state.time >= (self.end_time - fenics.dolfin.DOLFIN_EPS):
                    
                        phaseflow.helpers.print_once("Reached end time, t = " + str(self.end_time))
                    
                        break
                    
                    
                # Set initial values for next time step.
                self.old_state.solution.leaf_node().vector()[:] = self.state.solution.leaf_node().vector()
                
                self.old_state.time = 0. + self.state.time
    
        
    def set_unsteadiness(self):
    
        time_residual = fenics.Function(self.state.solution.leaf_node().function_space())
        
        time_residual.assign(self.state.solution.leaf_node() - self.old_state.solution.leaf_node())
        
        self.unsteadiness = fenics.norm(time_residual, "L2")/ \
            fenics.norm(self.old_state.solution.leaf_node(), "L2")
        
        phaseflow.helpers.print_once(
            "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = " + 
                str(self.unsteadiness))
                
                
    def write_checkpoint(self):
    
        pass
