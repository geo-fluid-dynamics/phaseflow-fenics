"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers
import os


class SolutionFile(fenics.XDMFFile):

    def __init__(self, filepath):

        phaseflow.helpers.mkdir_p(os.path.dirname(filepath))
        
        fenics.XDMFFile.__init__(self, filepath)
        
        self.parameters["functions_share_mesh"] = True  
        
        self.path = filepath
    
    
class State:

    def __init__(self, function_space, time = 0.):
    
        self.solution = fenics.Function(function_space)
        
        self.time = time
        
    
    def write_solution_to_xdmf(self, file):
        """Write the solution to a file.
        
        Parameters
        ----------
        file : fenics.XDMFFile
        
            write_solution should have been called from within the context of this open fenics.XDMFFile.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))
        
        pressure, velocity, temperature = self.solution.leaf_node().split()
    
        pressure.rename("p", "pressure")
        
        velocity.rename("u", "velocity")
        
        temperature.rename("T", "temperature")
        
        for var in [pressure, velocity, temperature]:
        
            file.write(var, self.time)
        
    
    def write_checkpoint(self, output_dir):
        """Write the checkpoint file (with solution and time)."""
        phaseflow.helpers.makedir_p(output_dir)
        
        filepath = output_dir + "/checkpoint_t" + str(self.time) + ".h5"
         
        phaseflow.helpers.print_once("Writing checkpoint file to " + filepath)
        
        with fenics.HDF5File(fenics.mpi_comm_world(), filepath, "w") as h5:
                    
            h5.write(self.solution.function_space().mesh().leaf_node(), "mesh")
        
            h5.write(self.solution.leaf_node(), "solution")
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with h5py.File(filepath, "r+") as h5:
                
                h5.create_dataset("time", data=self.time)
        
        
    def read_checkpoint(self, filepath):
        """Read the checkpoint file (with solution and time)."""
        phaseflow.helpers.print_once("Reading checkpoint file from " + filepath)
        
        mesh = fenics.Mesh()
            
        with fenics.HDF5File(mesh.mpi_comm(), filepath, "r") as h5:
        
            h5.read(mesh, "mesh", True)
        
        function_space = fenics.FunctionSpace(mesh, self.solution.function_space().element())

        self.solution = fenics.Function(function_space)

        with fenics.HDF5File(mesh.mpi_comm(), filepath, "r") as h5:
        
            h5.read(self.solution, "solution")
            
        with h5py.File(filepath, "r") as h5:
                
            self.time = h5["time"].value
        
    
class Model:
    
    def __init__(self,
            mesh, 
            element,
            initial_values,
            boundary_conditions = None, 
            timestep_bounds = (1.e-4, 1., 1.e12),
            quadrature_degree = None):
        """
        Parameters
        ----------
        model : phaseflow.Model
        velocity_boundary_conditions : {str: (float,),}
        temperature_boundary_conditions : {str: float,}
        timestep_bounds : float or (float, float, float)
        """
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fenics.FunctionSpace(mesh, element)
        
        self.state = State(self.function_space)

        self.old_state = State(self.function_space)
        
        self.old_state.solution = fenics.interpolate(
            fenics.Expression(initial_values, element = element), 
            self.function_space)
        
        if type(timestep_bounds) == type(1.):
        
            timestep_bounds = (timestep_bounds, timestep_bounds, timestep_bounds)
            
        self.timestep_size = TimeStepSize(
            min = timestep_bounds[0], 
            value = timestep_bounds[1], 
            max = timestep_bounds[2])
            
        self.Delta_t = fenics.Constant(self.timestep_size.value)
        
        self.variational_form = None
        
        self.derivative_of_variational_form = None
        
        self.problem = None
        
        if quadrature_degree is None:
        
            self.integration_metric = fenics.dx

        else:
        
            self.integration_metric = fenics.dx(metadata={'quadrature_degree': quadrature_degree})
        
        self.boundary_conditions = []
        
        for bc in boundary_conditions:
        
            self.boundary_conditions.append(
                fenics.DirichletBC(self.function_space.sub(bc["subspace"]), 
                    bc["value"], 
                    bc["location"]))
        
        
    def setup_problem(self):
        """ This must be called after setting the variational form and its derivative. """
        self.problem = fenics.NonlinearVariationalProblem( 
            self.variational_form, 
            self.state.solution, 
            self.boundary_conditions, 
            self.derivative_of_variational_form)
        
        
    def set_timestep_size_value(self, value):
    
        self.timestep_size.set(value)
                        
        self.Delta_t.assign(self.timestep_size.value)
        
        
    def write_dict(self):
    
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            file = open(output_dir + "/problem_dict.txt", "w")
            
            file.write(self.__dict__)

            file.close()
        
        
class Solver():

    def __init__(self, 
            model, 
            adaptive_goal_integrand = None, 
            adaptive_solver_tolerance = 1.e-4,
            initial_guess = None,
            nlp_max_iterations = 50,
            nlp_absolute_tolerance = 1.e-10,
            nlp_relative_tolerance = 1.e-9,
            nlp_relaxation = 1.):
    
        self.adaptive_goal_integrand = adaptive_goal_integrand
        
        if adaptive_goal_integrand is not None:
        
            self.adaptive_solver_tolerance = adaptive_solver_tolerance
    
            dx = model.integration_metric
        
            self.fenics_solver = fenics.AdaptiveNonlinearVariationalSolver(
                problem = model.problem,
                goal = adaptive_goal_integrand*dx)
                
            self.fenics_solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
                ["maximum_iterations"] = nlp_max_iterations
        
            self.fenics_solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
                ["absolute_tolerance"] = nlp_absolute_tolerance
            
            self.fenics_solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
                ["relative_tolerance"] = nlp_relative_tolerance
            
            self.fenics_solver.parameters["nonlinear_variational_solver"]["newton_solver"]\
                ["relaxation_parameter"] = nlp_relaxation
        
        else:
        
            self.fenics_solver = fenics.NonlinearVariationalSolver(problem = model.problem)
            
            self.fenics_solver.parameters["newton_solver"]["maximum_iterations"] = nlp_max_iterations
        
            self.fenics_solver.parameters["newton_solver"]["absolute_tolerance"] = nlp_absolute_tolerance
            
            self.fenics_solver.parameters["newton_solver"]["relative_tolerance"] = nlp_relative_tolerance
            
            self.fenics_solver.parameters["newton_solver"]["relaxation_parameter"] = nlp_relaxation
            
        if initial_guess is None:
        
            model.state.solution.leaf_node().vector()[:] = \
                model.old_state.solution.leaf_node().vector()
        
        else:
        
            initial_guess_function = fenics.interpolate(
                fenics.Expression(initial_guess, element = model.element), 
                model.function_space.leaf_node())
                
            model.state.solution.leaf_node().vector()[:] = \
                initial_guess_function.leaf_node().vector()
    
    
    def solve(self):
    
        if self.adaptive_goal_integrand is not None:
    
            self.fenics_solver.solve(self.adaptive_solver_tolerance)
            
        else:
        
            self.fenics_solver.solve()
    
    
class TimeStepper:

    time_epsilon = 1.e-8
    
    def __init__(self,
            model,
            solver,
            time_epsilon = time_epsilon,
            max_timesteps = 1000000000000,
            output_dir = None,
            stop_when_steady = False,
            steady_relative_tolerance = 1.e-4,
            adapt_timestep_to_unsteadiness = False,
            adaptive_time_power = 1.,
            end_time = None): 
    
        self.time_epsilon = time_epsilon

        self.max_timesteps = max_timesteps
        
        self.solver = solver
        
        self.model = model
        
        self.state = model.state
        
        self.old_state = model.old_state
        
        self.timestep_size = model.timestep_size

        self.output_dir = output_dir
        
        self.solution_file = None
        
        self.stop_when_steady = stop_when_steady
        
        self.steady_relative_tolerance = steady_relative_tolerance
        
        self.adapt_timestep_to_unsteadiness = adapt_timestep_to_unsteadiness
        
        self.adaptive_time_power = adaptive_time_power
        
        self.end_time = end_time
        
        
    def run_until_end_time(self):
        """ Optionally run inside of a file context manager.
        Without this, exceptions are more likely to corrupt the outputs.
        """
        if self.output_dir is None:
        
            self.__run_until_end_time()
        
        else:
        
            solution_filepath = self.output_dir + "/solution.xdmf"
        
            with SolutionFile(solution_filepath) as self.solution_file:
            
                self.old_state.write_solution_to_xdmf(self.solution_file)
            
                self.__run_until_end_time()
                
        
    def __run_until_end_time(self):

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
            
            self.solver.solve()
        
            self.state.time += self.timestep_size.value
    
            phaseflow.helpers.print_once("Reached time t = " + str(self.state.time))
            
            if self.output_dir is not None:
            
                self.state.write_solution_to_xdmf(self.solution_file)

                self.state.write_checkpoint(output_dir = self.output_dir)
            
            
            # Check for steady state.
            if self.stop_when_steady:
            
                self.set_unsteadiness()
                
                if (self.unsteadiness < self.steady_relative_tolerance):
            
                    steady = True
                    
                    phaseflow.helpers.print_once("Reached steady state at time t = " + str(self.state.time))
                    
                    break
                    
                if self.adapt_timestep_to_unsteadiness:

                    new_timestep_size = \
                        self.timestep_size.value/self.unsteadiness**self.adaptive_time_power
                    
                    self.model.set_timestep_size_value(new_timestep_size)
                        
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
        

class ContinuousFunction:

    def __init__(self, function, derivative_function):
    
        self.function = function
        
        self.derivative_function = derivative_function
      
        
class Point(fenics.Point):

    def __init__(self, coordinates):
    
        if type(coordinates) is type(0.):
        
            coordinates = (coordinates,)
        
        if len(coordinates) == 1:
        
            fenics.Point.__init__(self, coordinates[0])
            
        elif len(coordinates) == 2:
        
            fenics.Point.__init__(self, coordinates[0], coordinates[1])
            
        elif len(coordinates) == 3:
        
            fenics.Point.__init__(self, coordinates[0], coordinates[1], coordinates[2])
            
            

class BoundedValue(object):

    def __init__(self, min=0., value=0., max=0.):
    
        self.min = min
        
        self.value = value
        
        self.max = max
        
    
    def set(self, value):
    
        if value > self.max:
        
            value = self.max
            
        elif value < self.min:
        
            value = self.min
            
        self.value = value
        
        
class TimeStepSize(BoundedValue):
    """This class implements a bounded adaptive time step size."""
    def set(self, value):

        old_value = 0. + self.value
        
        BoundedValue.set(self, value)
        
        if abs(self.value - old_value) > TimeStepper.time_epsilon:
        
            phaseflow.helpers.print_once("Changed time step size from " + str(old_value) +
                " to " + str(self.value))
            
            
if __name__=="__main__":

    pass
    