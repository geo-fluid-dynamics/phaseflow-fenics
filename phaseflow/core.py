"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers


class SolutionFile(fenics.XDMFFile)

    def __init__(self, filepath):
    
        fenics.XDMFFile.__init__(self, filepath)
        
        self.path = filepath
    
    
class State():

    def __init__(self, time = 0., function_space):
    
        self.solution = fenics.Function.(function_space)
        
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
        
        for i, var in enumerate([pressure, velocity, temperature]):
        
            file.write(var, time)
        
    
    def write_checkpoint(self, output_dir):
        """Write the checkpoint file (with solution and time)."""
        filepath = output_dir + "/checkpoint_t" + str(self.time) + ".h5"
         
        phaseflow.helpers.print_once("Writing checkpoint file to " + filepath)
        
        with fenics.HDF5File(fenics.mpi_comm_world(), filepath, "w") as h5:
                    
            h5.write(self.solution.function_space().mesh().leaf_node(), "mesh")
        
            h5.write(self.solution.leaf_node(), "solution")
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with h5py.File(filepath, "r+") as h5:
                
                h5.create_dataset("time", data=time)
        
        
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
        
    
class Model():
    
    def __init__(mesh, element, quadrature_degree = None):
    
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fenics.FunctionSpace(mesh, element)
        
        self.state = State(function_space)
        
        self.old_state = State(function_space)
        
        self.time_step_size = 1.
        
        self.variational_form = None
        
        self.derivative_of_variational_form = None
        
        if quadrature_degree is None:
        
            self.integration_metric = fenics.dx

        else:
        
            self.integration_metric = fenics.dx(metadata={'quadrature_degree': quadrature_degree}
    
            
class Problem(fenics.NonlinearVariationalProblem):

    def __init__(self, 
            model, 
            velocity_boundary_conditions = None, 
            temperature_boundary_conditions = None)
        """
        Parameters
        ----------
        model : phaseflow.Model
        velocity_boundary_conditions : {str: (float,),}
        temperature_boundary_conditions : {str: float,}
        time_step_size : float
        """
        boundary_conditions = []
        
        for bc in velocity_boundary_conditions:
        
            boundary_conditions.append(
                fenics.DirichletBC(model.function_space.sub(model.element.velocity_subspace), 
                    bc.value, 
                    bc.location))
                    
        for bc in temperature_boundary_conditions:
        
            boundary_conditions.append(
                fenics.DirichletBC(model.function_space.sub(model.element.temperature_subspace), 
                    bc.value, 
                    bc.location))
                    
        fenics.NonlinearVariationalProblem.__init__(self, 
            model.nonlinear_variational_form, 
            model.state.solution, 
            boundary_conditions, 
            model.gateaux_derivative)
            
        self.model = model
        
    
    def write_dict(self):
    
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            file = open(output_dir + "/problem_dict.txt", "w")
            
            file.write(self.__dict__)

            file.close()
        
        
class Solver(fenics.AdaptiveNonlinearVariationalSolver):

    def __init__(self, problem, adaptive_goal_functional, adaptive_solver_tolerance = 1.e-4):
    
        fenics.AdaptiveNonlinearVariationalSolver.__init__(self, 
            problem = problem,
            goal = adaptive_goal_functional)
            
        self.problem = problem
        
        self.adaptive_solver_tolerance = adaptive_solver_tolerance
        
        
    def solve():

        fenics.AdaptiveNonlinearVariationalSolver.solve(self, self.adaptive_solver_tolerance)
    
    
class TimeStepper():

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
        
        self.model = solver.problem.model

        self.output_dir = output_dir
        
        self.stop_when_steady = stop_when_steady
        
        self.steady_relative_tolerance = steady_relative_tolerance
        
        
    def run_until(end_time):
    
        start_time = 0. + time
    
    
        # Open a context manager for the output file.
        solution_filepath = output_dir + "/solution.xdmf"
        
        with fenics.XDMFFile(solution_filepath) as solution_file:

        
            # Write the initial values.
            self.model.old_state.write_solution_to_xdmf(
                solution_file, 
                self.model.old_state.solution,
                self.model.old_state.time,
                solution_filepath) 

            if start_time >= end_time - self.time_epsilon:
        
                phaseflow.helpers.print_once(
                    "Start time is already too close to end time. Only writing initial values.")
                
                return
                
                
            # Run solver for each time step until reaching end time.
            progress = fenics.Progress("Time-stepping")
            
            fenics.set_log_level(fenics.PROGRESS)
            
            for it in range(1, self.max_time_steps):
                
                if(self.model.state.time > end_time - self.time_epsilon):
                    
                    break
                    
                self.run_time_step(problem)
        
                phaseflow.helpers.print_once("Reached time t = " + str(self.model.state.time))
                
                self.model.state.write_solution(solution_file)
                
                
                # Write checkpoint file.
                model.state.write_checkpoint(output_dir = self.output_dir)
                
                
                # Check for steady state.
                if self.stop_when_steady and self.steady():
                
                    phaseflow.helpers.print_once("Reached steady state at time t = " + str(time))
                    
                    break
                
                
                # Report progress.
                progress.update(time / end_time)
                
                if time >= (end_time - fenics.dolfin.DOLFIN_EPS):
                
                    phaseflow.helpers.print_once("Reached end time, t = " + str(end_time))
                
                    break
        
    
    def run_time_step(problem):

        self.old_state.solution.leaf_node().vector()[:] = self.state.solution.leaf_node().vector()
        
        self.old_state.time = self.state.time
        
        self.solver.solve()
        
        self.model.state.time += self.model.time_step_size
        
        
    def steady():
        """Check if solution has reached an approximately steady state."""
        steady = False
        
        time_residual = fenics.Function(self.model.state.solution.function_space().)
        
        time_residual.assign(self.model.state.solution - self.model.old_state.solution)
        
        unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(self.model.old_state.solution, "L2")
        
        phaseflow.helpers.print_once(
            "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = " + 
                str(unsteadiness))

        if (unsteadiness < self.steady_relative_tolerance):
            
            steady = True
        
        return steady
        

class ContinuousFunction():

    def __init__(self, function, derivative_function)
    
        self.function = function
        
        self.derivative_function = derivative_function
      
                
if __name__=="__main__":

    pass
    