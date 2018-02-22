"""**core.py** contains the core functionality."""
import fenics
import h5py
import numpy
import phaseflow.helpers
import tempfile
import pickle


class SolutionFile(fenics.XDMFFile):

    def __init__(self, filepath):

        fenics.XDMFFile.__init__(self, filepath)
        
        self.parameters["functions_share_mesh"] = True  
        
        self.path = filepath
    
    
class State:

    def __init__(self, function_space, element, time = 0.):
        """ **State** collects the solution and associated time.
        
        Parameters
        ----------
        function_space : fenics.FunctionSpace
        
        element : fenics.MixedElement
        
            Referencing the element here is necessary for the read_checkpoint method,
            since the type of function_space
            
        time : float
        """
        self.solution = fenics.Function(function_space)
        
        self.time = time
        
        self.function_space = function_space
        
        self.element = element
        
    
    def interpolate(self, expression_strings):
        """Interpolate the solution from mathematical expressions. """
        interpolated_solution = fenics.interpolate(
            fenics.Expression(expression_strings, element = self.element), 
            self.function_space.leaf_node())
        
        self.solution.leaf_node().vector()[:] = interpolated_solution.leaf_node().vector() 
        
    
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
        
    
class WeakForm:

    def __init__(self, solutions, integration_metric):
    
        self.setup(solutions, integration_metric)
        
        
    def setup(self):
        """ This must be overloaded. """
        assert(False)
    
    
class Model:
    
    def __init__(self,
            mesh, 
            element,
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
        
        self.boundary_condition_dicts = boundary_conditions
        
        self.boundary_conditions = []
        
        self.function_space = fenics.FunctionSpace(self.mesh, self.element)
        
        self.state = State(self.function_space, self.element)

        self.old_state = State(self.function_space, self.element)
        
        self.setup_problem()
        
        self.setup_adaptive_goal_integrand()
        
        
    def setup_problem(self):
        
        self.setup_variational_form()
        
        if self.derivative_of_variational_form is None:
        
            self.derivative_of_variational_form = fenics.derivative(self.variational_form, 
                self.state.solution, 
                fenics.TrialFunction(self.function_space))
        
        boundary_conditions = []
        
        for dict in self.boundary_condition_dicts:
        
            boundary_conditions.append(
                fenics.DirichletBC(self.function_space.sub(dict["subspace"]), 
                    dict["value"], 
                    dict["location"]))
                    
        self.problem = fenics.NonlinearVariationalProblem( 
            self.variational_form, 
            self.state.solution, 
            boundary_conditions, 
            self.derivative_of_variational_form)
    

    def setup_adaptive_goal_integrand(self):
        """ This must be overloaded for adaptive problems. """
        pass
    
    
    def setup_variational_form(self):
        """ Define the fenics.NonlinearVariationalForm.
        
        The derived Model class must overload this method. """
        assert(False)
    
    
    def set_timestep_size_value(self, value):
        """ Set the value of the time step size."""
        self.timestep_size.set(value)
                        
        self.Delta_t.assign(self.timestep_size.value)
        
    
    def write_checkpoint(self, output_dir):
        """Write the checkpoint file (with solution and time for the latest state)."""        
        filepath = output_dir + "/checkpoint_t" + str(self.state.time) + ".h5"
         
        phaseflow.helpers.print_once("Writing checkpoint file to " + filepath)
        
        with fenics.HDF5File(fenics.mpi_comm_world(), filepath, "w") as h5:
                    
            h5.write(self.state.solution.function_space().mesh().leaf_node(), "mesh")
        
            h5.write(self.state.solution.leaf_node(), "solution")
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with h5py.File(filepath, "r+") as h5:
                
                h5.create_dataset("time", data=self.state.time)

        
    def read_checkpoint(self, filepath):
        """Read the checkpoint file (with solution and time)."""
        phaseflow.helpers.print_once("Reading checkpoint file from " + filepath)
        
        self.mesh = fenics.Mesh()
            
        with fenics.HDF5File(self.mesh.mpi_comm(), filepath, "r") as h5:
        
            h5.read(self.mesh, "mesh", True)
        
        self.function_space = fenics.FunctionSpace(self.mesh, self.element)

        self.old_state.solution = fenics.Function(self.function_space)
        
        self.state.solution = fenics.Function(self.function_space)

        with fenics.HDF5File(self.mesh.mpi_comm(), filepath, "r") as h5:
        
            h5.read(self.old_state.solution, "solution")
            
        with h5py.File(filepath, "r") as h5:
                
            self.old_state.time = h5["time"].value
            
        self.state.solution.vector()[:] = self.old_state.solution.vector()
        
        self.state.time = 0. + self.old_state.time
            
        self.setup_problem()
        

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
            output_dir_suffix = None,
            prefix_output_dir_with_tempdir = False,
            stop_when_steady = False,
            steady_relative_tolerance = 1.e-4,
            adapt_timestep_to_unsteadiness = False,
            adaptive_time_power = 1.,
            end_time = None): 
        """
        Parameters
        ----------
        prefix_output_dir_with_tempdir : bool
        
            Sometimes it's necessary to write to a safe temporary directory, e.g. for Travis-CI.
        """
        self.time_epsilon = time_epsilon

        self.max_timesteps = max_timesteps
        
        self.solver = solver
        
        self.model = model
        
        self.state = model.state
        
        self.old_state = model.old_state
        
        self.timestep_size = model.timestep_size

        self.output_dir = "phaseflow/output/" + output_dir_suffix
        
        if prefix_output_dir_with_tempdir:
        
            self.output_dir = tempfile.mkdtemp() + "/" + self.output_dir
        
        self.solution_file = None
        
        self.stop_when_steady = stop_when_steady
        
        self.steady_relative_tolerance = steady_relative_tolerance
        
        self.adapt_timestep_to_unsteadiness = adapt_timestep_to_unsteadiness
        
        self.adaptive_time_power = adaptive_time_power
        
        self.end_time = end_time
        
        
    def run_until_end_time(self):
        
        solution_filepath = self.output_dir + "/solution.xdmf"
    
        with SolutionFile(solution_filepath) as self.solution_file:
            """ Run inside of a file context manager.
            Without this, exceptions are more likely to corrupt the outputs.
            """
            self.old_state.write_solution_to_xdmf(self.solution_file)
        
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
                
                self.state.write_solution_to_xdmf(self.solution_file)

                self.model.write_checkpoint(output_dir = self.output_dir)
                
                
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
        

class Simulation():

    def __init__(model, timestepper):
    
        self.model = model
        
        self.timestepper = timestepper
        
        self.adaptive_goal_integrand = None
        
        self.adaptive_solver_tolerance = 1.e-4
        
        self.output_dir_suffix = "benchmarks/"
        
        self.output_dir = None
        
        self.end_time = None
        
        self.stop_when_steady = False
        
        self.steady_relative_tolerance = 1.e-4
        
        self.adapt_timestep_to_unsteadiness = False
        
        self.adaptive_time_power = 1.
        
        self.initial_guess = None
        
        self.prefix_output_dir_with_tempdir = False
        
    
    def run(self):
    
        solver = phaseflow.core.Solver(
            model = self.model, 
            adaptive_goal_integrand = self.adaptive_goal_integrand, 
            adaptive_solver_tolerance = self.adaptive_solver_tolerance,
            initial_guess = self.initial_guess)
        
        timestepper = phaseflow.core.TimeStepper(
            model = self.model,
            solver = solver,
            output_dir_suffix = self.output_dir_suffix,
            prefix_output_dir_with_tempdir = self.prefix_output_dir_with_tempdir,
            end_time = self.end_time,
            stop_when_steady = self.stop_when_steady,
            steady_relative_tolerance = self.steady_relative_tolerance,
            adapt_timestep_to_unsteadiness = self.adapt_timestep_to_unsteadiness,
            adaptive_time_power = self.adaptive_time_power)
        
        self.output_dir = timestepper.output_dir
        
        timestepper.run_until_end_time()
        
        
        
class ContinuousFunction:

    def __init__(self, function, derivative_function):
    
        self.function = function
        
        self.derivative_function = derivative_function
        
    
    def __call__(self, arg):
        
        return self.function(arg)
      
        
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
    