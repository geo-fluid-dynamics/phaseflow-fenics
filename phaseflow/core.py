"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers


class SolutionFile(fenics.XDMFFile):

    def __init__(self, filepath):
    
        fenics.XDMFFile.__init__(self, filepath)
        
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
            initial_values = None,
            boundary_conditions = None, 
            time_step_size = 1.,
            quadrature_degree = None):
        """
        Parameters
        ----------
        model : phaseflow.Model
        velocity_boundary_conditions : {str: (float,),}
        temperature_boundary_conditions : {str: float,}
        time_step_size : float
        """
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fenics.FunctionSpace(mesh, element)
        
        self.state = State(self.function_space)
        
        if initial_values is not None:
        
            self.state.solution = fenics.interpolate(
                fenics.Expression(initial_values, element = element), 
                self.function_space)
        
        self.old_state = State(self.function_space)
        
        self.old_state.solution.leaf_node().vector()[:] = self.state.solution.leaf_node().vector()
        
        self.time_step_size = time_step_size
        
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
        
        
    def write_dict(self):
    
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            file = open(output_dir + "/problem_dict.txt", "w")
            
            file.write(self.__dict__)

            file.close()
        
        
class Solver():

    def __init__(self, model, adaptive_goal_integrand, adaptive_solver_tolerance = 1.e-4):
    
        self.model = model
        
        self.adaptive = adaptive_goal_integrand is not None

        if self.adaptive:
    
            self.adaptive_solver_tolerance = adaptive_solver_tolerance
            
            dx = model.integration_metric
            
            self.solver = fenics.AdaptiveNonlinearVariationalSolver(
                problem = model.problem,
                goal = adaptive_goal_integrand*dx)
        
        else:
        
            self.solver = fenics.NonlinearVariationalSolver(problem = model.problem)
        
        
    def solve(self):

        if self.adaptive:
        
            self.solver.solve(self, self.adaptive_solver_tolerance)
            
        else:
        
            self.solver.solve()
    
    
class TimeStepper:

    def __init__(self, 
            solver,
            time_epsilon = 1.e-8,
            max_time_steps = 1000000000000,
            output_dir = None,
            stop_when_steady = False,
            steady_relative_tolerance = 1.e-4): 
    
        self.time_epsilon = time_epsilon

        self.max_time_steps = max_time_steps
        
        self.solver = solver
        
        self.state = solver.model.state
        
        self.old_state = solver.model.old_state

        self.output_dir = output_dir
        
        self.solution_file = None
        
        self.stop_when_steady = stop_when_steady
        
        self.steady_relative_tolerance = steady_relative_tolerance
        
        
    def run_until(self, end_time):
        """ Optionally run inside of a file context manager.
        Without this, exceptions are more likely to corrupt the outputs.
        """
        if self.output_dir is None:
        
            self.__run_until(end_time)
        
        else:
        
            solution_filepath = self.output_dir + "/solution.xdmf"
        
            with SolutionFile(solution_filepath) as self.solution_file:
            
                self.solver.model.state.write_solution_to_xdmf(self.solution_file)
            
                self.__run_until(end_time)
                
        
    def __run_until(self, end_time):

        start_time = 0. + self.state.time

        if start_time >= end_time - self.time_epsilon:
    
            phaseflow.helpers.print_once(
                "Start time is already too close to end time. Only writing initial values.")
            
            return
            
            
        # Run solver for each time step until reaching end time.
        progress = fenics.Progress("Time-stepping")
        
        fenics.set_log_level(fenics.PROGRESS)
        
        for it in range(1, self.max_time_steps):
            
            if(self.state.time > end_time - self.time_epsilon):
                
                break
                
            self.run_time_step()
    
            phaseflow.helpers.print_once("Reached time t = " + str(self.state.time))
            
            if self.output_dir is not None:
            
                self.state.write_solution_to_xdmf(self.solution_file)

                self.state.write_checkpoint(output_dir = self.output_dir)
            
            
            # Check for steady state.
            if self.stop_when_steady and self.steady():
            
                phaseflow.helpers.print_once("Reached steady state at time t = " + str(self.state.time))
                
                break
            
            
            # Report progress.
            progress.update(self.state.time / end_time)
            
            if self.state.time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                phaseflow.helpers.print_once("Reached end time, t = " + str(end_time))
            
                break
    
    
    def run_time_step(self):

        self.old_state.solution.leaf_node().vector()[:] = self.state.solution.leaf_node().vector()
        
        self.old_state.time = self.state.time
        
        self.solver.solve()
        
        self.state.time += self.solver.model.time_step_size
        
        
    def steady(self):
        """Check if solution has reached an approximately steady state."""
        steady = False
        
        time_residual = fenics.Function(self.state.solution.function_space())
        
        time_residual.assign(self.state.solution - self.old_state.solution)
        
        unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(self.old_state.solution, "L2")
        
        phaseflow.helpers.print_once(
            "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = " + 
                str(unsteadiness))

        if (unsteadiness < self.steady_relative_tolerance):
            
            steady = True
        
        return steady
        

class ContinuousFunction:

    def __init__(self, function, derivative_function):
    
        self.function = function
        
        self.derivative_function = derivative_function
      
                
if __name__=="__main__":

    pass
    