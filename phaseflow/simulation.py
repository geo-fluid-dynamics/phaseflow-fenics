""" **simulation.py** contains the Simulation class. """
import fenics
import phaseflow
import tempfile
import pprint
import h5py


class Simulation:
    """ This is a 'god class' which acts as an API for writing Phaseflow models and applications. 
    
    For an example model, see `phaseflow.octadecane`, 
    with corresponding example applications in `phaseflow.octadecane_benchmarks`.
    """
    def __init__(self):
        """ Initialize attributes which should be modified by the user before calling `self.run`."""
        
        
        # Adaptive FEM solver
        self.adaptive_goal_tolerance = 1.e12
        
        
        # Nonlinear solver
        self.nonlinear_solver_max_iterations = 50
        
        self.nonlinear_solver_absolute_tolerance = 1.e-10
        
        self.nonlinear_solver_relative_tolerance = 1.e-9
        
        self.nonlinear_solver_relaxation = 1.
        
        
        # Time integration
        self.end_time = 1.
        
        self.timestep_size = 1.
        
        self.second_order_time_discretization = False
        
        self.stop_when_steady = False
        
        self.steady_relative_tolerance = 1.e-4
        
        
        # File output
        self.output_dir = "phaseflow/output/"
        
        self.prefix_output_dir_with_tempdir = False
        
        
        # Quadrature
        """ The degree of the quadrature rule used for numerical integration. 
        
        If `self.quadrature_degree = None`, then the exact quadrature rule will be used.
        
        Many of the benchmark tests use the exact quadrature rule.
        In some cases, setting a lower degree, e.g. `self.quadrature_degree = 8`, leads to great performance benefits.
        """
        self.quadrature_degree = None
        
        
        # Grid coarsening
        self.coarsen_between_timesteps = False
        
        self.coarsening_absolute_tolerance = 1.e-3
        
        self.coarsening_maximum_refinement_cycles = 6
        
        self.coarsening_scalar_solution_component_index = 3
        
        
        #
        self.init_hidden_attributes()
        
        
    def setup_coarse_mesh(self):
        """ This must be overloaded to instantiate a `fenics.Mesh` at `self.mesh`. """
        raise(NotImplementedError())
    

    def setup_element(self):
        """ This must be overloaded to instantiate a `fenics.MixedElement` at `self.element`. """
        raise(NotImplementedError())
    
    
    def setup_initial_values(self):
        """ This must be overloaded to set `self.old_state.solution`. 
        
        Often this might involve calling the `self.old_state.interpolate` method.
        """
        raise(NotImplementedError())
        
        
    def setup_governing_form(self):
        """ Set the variational form for the governing equations.
        
        This must be overloaded.
        
        Optionally, self.derivative_of_governing_form can be set here.
        Otherwise, the derivative will be computed automatically.
        """
        raise(NotImplementedError())
        
    
    def refine_initial_mesh(self):
        """ Overload this to refine the mesh before adaptive mesh refinement. """
        pass
        
        
    def validate_attributes(self):
        """ Overload this to validate attributes set by the user. 
        
        The goal should be to improve user friendliness, or otherwise reduce lines of user code.
        
        For example, phaseflow.octadecane_benchmarks.CavityBenchmarkSimulation overloads
        .. code-block::python
        
            def validate_attributes(self):
    
                if type(self.mesh_size) is type(20):
                
                    self.mesh_size = (self.mesh_size, self.mesh_size)
        
        
        since the domain is often a unit square discretized uniformly in both directions.
        """
        pass
        
        
    def init_hidden_attributes(self):
        """ Initialize attributes which should be hidden from the user. """
        self.restarted = False
        
        self.timestep = 0
        
        self.max_timesteps = 1000000000000
        
        self.time_epsilon = 1.e-8
        
        self.minimum_timestep_size = 1.e-4
        
        self.maximum_timestep_size = 1.e12
        
        self.fenics_timestep_size = fenics.Constant(self.timestep_size)
        
       
    def setup(self):
        """ Set up objects needed before the simulation can run. """
        self.validate_attributes()
        
        self.setup_derived_attributes()
        
        if not self.restarted:
            
            self.setup_coarse_mesh()
            
            self.setup_element()
            
            self.refine_initial_mesh()
            
            self.setup_function_space()
            
            self.setup_states()
            
            self.setup_initial_values()
            
            if self.second_order_time_discretization:
            
                self.old_old_state.set_from_other_state(self.old_state)
            
                self.old_old_state.time -= self.timestep_size
        
        self.setup_problem_and_solver()
        
        if self.prefix_output_dir_with_tempdir:
        
            self.output_dir = tempfile.mkdtemp() + "/" + self.output_dir
            
        phaseflow.helpers.mkdir_p(self.output_dir)
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with open(self.output_dir + '/simulation_vars.txt', 'w') as simulation_vars_file:
            
                pprint.pprint(vars(self), simulation_vars_file)
    
    
    def setup_problem_and_solver(self):
        """ Set up the `fenics.NonlinearVariationalProblem` and `fenics.AdaptiveNonlinearVariationalSolver`. """
        self.setup_governing_form()
        
        self.setup_boundary_conditions()
        
        self.setup_derivative()
        
        self.setup_problem()
        
        self.setup_adaptive_goal_form()
        
        self.setup_solver()
        
        self.setup_initial_guess()
    
    
    def setup_derived_attributes(self):
        """ Set attributes which shouldn't be touched by the user. """
        self.fenics_timestep_size = fenics.Constant(self.timestep_size)
        
        if self.second_order_time_discretization:
        
            self.old_fenics_timestep_size = fenics.Constant(self.timestep_size)
        
        if self.quadrature_degree is None:
        
            self.integration_metric = fenics.dx
        
        else:
        
            self.integration_metric = fenics.dx(metadata={'quadrature_degree': self.quadrature_degree})
        
    
    def setup_function_space(self):
        """ Set the function space. """
        self.function_space = fenics.FunctionSpace(self.mesh, self.element)
        
        
    def setup_states(self):    
        """ Set state objects which contain the solutions. """
        self.state = phaseflow.state.State(self.function_space, self.element)
        
        self.old_state = phaseflow.state.State(self.function_space, self.element)
        
        if self.second_order_time_discretization:
        
            self.old_old_state = phaseflow.state.State(self.function_space, self.element)
    
        
    def setup_derivative(self):
        """ Set the derivative of the governing form, needed for the nonlinear solver. """
        self.derivative_of_governing_form = fenics.derivative(self.governing_form, 
            self.state.solution, 
            fenics.TrialFunction(self.function_space))
        
        
    def setup_boundary_conditions(self):
        """ Set the collection of `fenics.DirichetBC` based on the user's provided collection 
            of boundary condition dictionaries.
            
        This format allows the user to specify boundary conditions abstractly,
        without referencing the actual `fenics.FunctionSpace` at `self.function_space`.
        """
        self.fenics_bcs = []
        
        for dict in self.boundary_conditions:
        
            self.fenics_bcs.append(
                fenics.DirichletBC(self.function_space.sub(dict["subspace"]), 
                    dict["value"], 
                    dict["location"]))
        

    def apply_time_discretization(self, Delta_t, u):
    
        u_t = phaseflow.backward_difference_formulas.apply_backward_euler(Delta_t, u)
        
        return u_t

        
    def setup_problem(self):
        """ Set the `fenics.NonlinearVariationalProblem`. """
        self.problem = fenics.NonlinearVariationalProblem( 
            self.governing_form, 
            self.state.solution, 
            self.fenics_bcs, 
            self.derivative_of_governing_form)
        
    
    def setup_adaptive_goal_form(self):
        """ Set the goal for adaptive mesh refinement.
        
        This should be overloaded for most applications.
        """
        self.adaptive_goal_form = self.state.solution[0]*self.integration_metric
        
        
    def setup_solver(self):
        """ Set up the solver, which is a `fenics.AdaptiveNonlinearVariationalSolver`. """
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

    
    def setup_initial_guess(self):
        """ Set the initial guess for the Newton solver.
        
        Using the latest solution as the initial guess should be fine for most applications.
        Otherwise, this must be overloaded.
        """
        self.state.set_solution_from_other_solution(self.old_state.solution)
        
    
    def run(self):
        """ Run the time-dependent simulation. 
        
        This is where everything comes together. As of this writing, this is the longest function 
        in Phaseflow. Not only does this contain the time loop, but it handles writing solution
        and checkpoint files, checks stopping criterion, and prints status messages.
        
        Eventually we may want to consider other time integration options,
        which will require redesigning this function.
        """
        if self.timestep == 0:

            self.setup()
        
        solution_filepath = self.output_dir + "/solution.xdmf"
    
        with phaseflow.helpers.SolutionFile(solution_filepath) as self.solution_file:
            """ Run inside of a file context manager.
            
            Without this, exceptions are more likely to corrupt the outputs.
            """
            if self.timestep == 0:
                
                self.do_before_timesteps()
            
            progress = fenics.Progress("Time-stepping")
            
            first_timestep = self.timestep + 1

            for self.timestep in range(first_timestep, self.max_timesteps):
                """ Run solver for each time step until reaching end time or max number of time steps. """
                if (self.end_time is not None) and (self.state.time > self.end_time - self.time_epsilon):
                    """ Check if the end time has been reached. """
                    phaseflow.helpers.print_once("Reached end time, t = " + str(self.end_time))
                    
                    break
                        
                if self.timestep > 1:
                    """ Handle some operations between time steps. """
                    self.do_between_timesteps()
                    
                    if self.coarsen_between_timesteps:
                    
                        self.coarsen()
                    
                self.state.time = self.old_state.time + self.timestep_size
                
                self.solver.solve(self.adaptive_goal_tolerance)
                
                self.write_solution(self.solution_file, self.state)
                
                self.write_checkpoint()
                
                if self.stop_when_steady:
                    """ Check for steady state. """
                    self.compute_unsteadiness()
                    
                    phaseflow.helpers.print_once("Unsteadiness = " + str(self.unsteadiness)
                        + " (Stopping at " + str(self.steady_relative_tolerance) + ")")
                        
                    if (self.unsteadiness < self.steady_relative_tolerance):
                        
                        phaseflow.helpers.print_once("Reached steady state at time t = " + str(self.state.time))
                        
                        break
                
                
    def do_before_timesteps(self):
    
        if self.second_order_time_discretization:
            
            self.write_solution(self.solution_file, self.old_old_state)
        
        self.write_solution(self.solution_file, self.old_state)
        
        fenics.set_log_level(fenics.PROGRESS)
            

    def do_between_timesteps(self):
        """ Handle some operations between time steps, mostly managing solutions at different times.
        
        Overload this with anything you want to be routinely done between time steps.
        
            For example: In the heat-driven cavity benchmark, 
            we keep doubling the time step size to quickly reach steady state.
        """
        phaseflow.helpers.print_once("Reached time t = " + str(self.state.time))
        
        if self.second_order_time_discretization:
                    
            self.old_old_state.set_from_other_state(self.old_state)
            
            self.old_fenics_timestep_size.assign(self.timestep_size)
        
        self.old_state.set_from_other_state(self.state)
        
        
    def compute_unsteadiness(self):
        """ Set 'unsteadiness' metric to compare to `steady_tolerance` for choosing to stop at steady state. """
        time_residual = fenics.Function(self.state.solution.leaf_node().function_space())
        
        time_residual.assign(self.state.solution.leaf_node() - self.old_state.solution.leaf_node())
        
        L2_norm_relative_time_residual = fenics.norm(time_residual.leaf_node(), "L2")/ \
            fenics.norm(self.old_state.solution.leaf_node(), "L2")
        
        self.unsteadiness = L2_norm_relative_time_residual

                
    def write_solution(self, file, state):
        """ Write the solution to a file.

        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile

            This method should have been called from within the context of the open `file`.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))

        for var in state.solution.leaf_node().split():

            file.write(var, state.time)

 
    def write_checkpoint(self):
        """Write checkpoint file (with solution and time) to disk."""
        checkpoint_filepath = self.output_dir + "checkpoint_t" + str(self.state.time) + ".h5"
        
        self.latest_checkpoint_filepath = checkpoint_filepath
        
        phaseflow.helpers.print_once("Writing checkpoint file to " + checkpoint_filepath)
        
        with fenics.HDF5File(fenics.mpi_comm_world(), checkpoint_filepath, "w") as h5:
            
            h5.write(self.state.solution.function_space().mesh().leaf_node(), "mesh")
        
            h5.write(self.state.solution.leaf_node(), "solution")
            
            if self.second_order_time_discretization:
            
                h5.write(self.old_state.solution.leaf_node(), "old_solution")
            
        if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
            with h5py.File(checkpoint_filepath, "r+") as h5:
                
                h5.create_dataset("time", data = self.state.time)
                
                h5.create_dataset("timestep_size", data = self.timestep_size)
                
                if self.second_order_time_discretization:
                
                    h5.create_dataset("old_time", data = self.old_state.time)
        
        
    def read_checkpoint(self, checkpoint_filepath):
        """Read the checkpoint solution and time, perhaps to restart."""
        phaseflow.helpers.print_once("Reading checkpoint file from " + checkpoint_filepath)
        
        self.mesh = fenics.Mesh()
            
        with fenics.HDF5File(self.mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
        
            h5.read(self.mesh, "mesh", True)
        
        self.setup_element()
            
        self.setup_function_space()
            
        self.setup_states()

        with fenics.HDF5File(self.mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
        
            h5.read(self.old_state.solution, "solution")
            
            if self.second_order_time_discretization:
            
                h5.read(self.old_old_state.solution, "old_solution")
            
        with h5py.File(checkpoint_filepath, "r") as h5:
                
            self.old_state.time = h5["time"].value
            
            self.set_timestep_size(h5["timestep_size"].value)
            
            if self.second_order_time_discretization:
            
                self.old_old_state.time = h5["old_time"].value
        
        self.restarted = True
        
        self.output_dir += "restarted_t" + str(self.old_state.time) + "/"
        
        
    def set_timestep_size(self, new_timestep_size):
        """ When using adaptive time stepping, this sets the time step size.
        
        This requires that `self.setup_governing_form` sets `self.fenics_timestep_size`,
        which must be a `fenics.Constant`. Given that, this calls the `fenics.Constant.assign` 
        method so that the change affects `self.governing_form`.
        """
        if new_timestep_size > self.maximum_timestep_size:
                        
            new_timestep_size = self.maximum_timestep_size
            
        if new_timestep_size < self.minimum_timestep_size:
        
            new_timestep_size = self.minimum_timestep_size

        self.timestep_size = 0. + new_timestep_size

        self.fenics_timestep_size.assign(new_timestep_size)
            
        if abs(new_timestep_size - self.timestep_size) > self.time_epsilon:
            
            print("Set the time step size to " + str(self.timestep_size))
        
        
    def coarsen(self):
        """ Remesh and refine the new mesh until the interpolation error tolerance is met. 
    
        For simplicity, for now we'll consider only one scalar component of the solution.
        """
        time = 0. + self.old_state.time
        
        fine_solution = self.state.solution.copy(deepcopy = True)
        
        self.setup_coarse_mesh()
        
        for refinement_cycle in range(self.coarsening_maximum_refinement_cycles):
            
            self.setup_function_space()
            
            self.setup_states()
            
            self.old_state.time = 0. + time
            
            self.old_state.solution = fenics.project(fine_solution.leaf_node(), self.function_space.leaf_node())
            
            if refinement_cycle == self.coarsening_maximum_refinement_cycles:
            
                break
                
            exceeds_tolerance = fenics.MeshFunction("bool", self.mesh.leaf_node(), self.mesh.topology().dim(), False)

            exceeds_tolerance.set_all(False)
        
            for cell in fenics.cells(self.mesh.leaf_node()):
                
                coarse_value = self.old_state.solution.leaf_node()(cell.midpoint())\
                    [self.coarsening_scalar_solution_component_index]
                    
                fine_value = fine_solution.leaf_node()(cell.midpoint())\
                    [self.coarsening_scalar_solution_component_index]
                
                if (abs(coarse_value - fine_value) > self.coarsening_absolute_tolerance):
                
                    exceeds_tolerance[cell] = True
                
            if any(exceeds_tolerance):
                
                self.mesh = fenics.refine(self.mesh, exceeds_tolerance)
                
            else:
            
                break
        
        self.setup_problem_and_solver()  # We broke important references.
        
