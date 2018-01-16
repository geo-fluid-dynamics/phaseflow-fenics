"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers

TIME_EPS = 1.e-8

pressure_degree = 1

temperature_degree = 1

""" The equations are scaled with unit Reynolds Number
per Equation 8 from danaila2014newton, i.e.

    v_ref = nu_liquid/H => t_ref = nu_liquid/H^2 => Re = 1.
"""
reynolds_number = 1.

MAX_TIME_STEPS = 1000000000000

def make_mixed_fe(cell):
    """ Define the mixed finite element.
    MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    """
    pressure_element = fenics.FiniteElement("P", cell, pressure_degree)
    
    velocity_degree = pressure_degree + 1
    
    velocity_element = fenics.VectorElement("P", cell, velocity_degree)

    temperature_element = fenics.FiniteElement("P", cell, temperature_degree)

    mixed_element = fenics.MixedElement([pressure_element, velocity_element, temperature_element])
    
    return mixed_element
  
  
def run(solution,
        initial_values,
        boundary_conditions,
        time = 0.,
        output_dir = "phaseflow_output/",
        rayleigh_number = 1.,
        prandtl_number = 1.,
        stefan_number = 1.,
        liquid_viscosity = 1.,
        solid_viscosity = 1.e8,
        gravity = (0., -1.),
        m_B = None,
        ddT_m_B = None,
        penalty_parameter = 1.e-7,
        semi_phasefield_mapping = None,
        semi_phasefield_mapping_derivative = None,
        end_time = 10.,
        time_step_size = 1.e-3,
        stop_when_steady = False,
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
    

    # Handle default function definitions.
    if m_B is None:
        
        def m_B(T, Ra, Pr, Re):
        
            return T*Ra/(Pr*Re**2)
    
    
    if ddT_m_B is None:
        
        def ddT_m_B(T, Ra, Pr, Re):

            return Ra/(Pr*Re**2)
    
    
    if semi_phasefield_mapping is None:
    
        assert (semi_phasefield_mapping_derivative is None)

        def semi_phasefield_mapping(T):
    
            return 0.
        
        def semi_phasefield_mapping_derivative(T):
        
            return 0.
            
        
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
    
    
    # Set the variational form.
    """Set local names for math operators to improve readability."""
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    """The linear, bilinear, and trilinear forms b, a, and c, follow the common notation 
    for applying the finite element method to the incompressible Navier-Stokes equations,
    e.g. from danaila2014newton and huerta2003fefluids.
    """
    def b(u, p):
        return -div(u)*p  # Divergence
    
    
    def D(u):
    
        return sym(grad(u))  # Symmetric part of velocity gradient
    
    
    def a(mu, u, v):
        
        return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    
    def c(u, z, v):
        
        return dot(dot(grad(z), u), v)  # Convection of the velocity field
    
    
    Delta_t = fenics.Constant(time_step_size)
    
    Re = fenics.Constant(reynolds_number)
    
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    Ste = fenics.Constant(stefan_number)
    
    g = fenics.Constant(gravity)
    
    def f_B(T):
    
        return m_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g  # Buoyancy force, $f = ma$
    
    
    phi = semi_phasefield_mapping
    
    gamma = fenics.Constant(penalty_parameter)
    
    mu_l = fenics.Constant(liquid_viscosity)
    
    mu_s = fenics.Constant(solid_viscosity)
    
    def mu(T):
    
        return mu_l + (mu_s - mu_l)*phi(T) # Variable viscosity.
    
    psi_p, psi_u, psi_T = fenics.TestFunctions(W)
    
    w = solution 
    
    p, u, T = fenics.split(w)
    
    w_n = initial_values
     
    p_n, u_n, T_n = fenics.split(w_n)
    
    if quadrature_degree is None:
    
        dx = fenics.dx
        
    else:
    
        dx = fenics.dx(metadata={'quadrature_degree': quadrature_degree})
    
    F = (
        b(u, psi_p) - psi_p*gamma*p
        + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
        + c(u, u, psi_u) + b(psi_u, p) + a(mu(T), u, psi_u)
        + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
        + dot(grad(psi_T), 1./Pr*grad(T) - T*u)        
        )*dx

        
    # Set the Jacobian (formally the Gateaux derivative) in variational form.
    def ddT_f_B(T):
        
        return ddT_m_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g
    
    dphi = semi_phasefield_mapping_derivative
    
    def dmu(T):
    
        return (mu_s - mu_l)*dphi(T)
        
        
    delta_w = fenics.TrialFunction(W)
    
    delta_p, delta_u, delta_T = fenics.split(delta_w)
    
    w_k = w
    
    p_k, u_k, T_k = fenics.split(w_k)
    
    JF = (
        b(delta_u, psi_p) - psi_p*gamma*delta_p 
        + dot(psi_u, 1./Delta_t*delta_u + delta_T*ddT_f_B(T_k))
        + c(u_k, delta_u, psi_u) + c(delta_u, u_k, psi_u) 
        + b(psi_u, delta_p) 
        + a(delta_T*dmu(T_k), u_k, psi_u) + a(mu(T_k), delta_u, psi_u) 
        + 1./Delta_t*psi_T*delta_T*(1. - 1./Ste*dphi(T_k))
        + dot(grad(psi_T), 1./Pr*grad(delta_T) - T_k*delta_u - delta_T*u_k)
        )*fenics.dx

        
    # Make the problem.
    problem = fenics.NonlinearVariationalProblem(F, w_k, boundary_conditions, JF)
    
    
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
    
    
def write_solution(solution_file, solution, time, solution_filepath):
    """Write the solution to disk.
    
    Parameters
    ----------
    solution_file : fenics.XDMFFile
    
        write_solution should have been called from within the context of an open fenics.XDMFFile.
    
    solution : fenics.Function
    
        The FEniCS function where the solution is stored.
    
    time : float
    
        The time corresponding to the time-dependent solution.
    
    solution_filepath : str
    
        This is needed because fenics.XDMFFile does not appear to have a method for providing the file path.
        With a Python file, one can simply do
        
            File = open("foo.txt", "w")
            
            File.name
            
        But fenics.XDMFFile.name returns a reference to something done with SWIG.
    
    """
    phaseflow.helpers.print_once("Writing solution to " + str(solution_filepath))
    
    pressure, velocity, temperature = solution.leaf_node().split()
    
    pressure.rename("p", "pressure")
    
    velocity.rename("u", "velocity")
    
    temperature.rename("T", "temperature")
    
    for i, var in enumerate([pressure, velocity, temperature]):
    
        solution_file.write(var, time)
        

def write_checkpoint(checkpoint_filepath, w, time):
    """Write checkpoint file (with solution and time) to disk."""
    phaseflow.helpers.print_once("Writing checkpoint file to " + checkpoint_filepath)
    
    with fenics.HDF5File(fenics.mpi_comm_world(), checkpoint_filepath, "w") as h5:
                
        h5.write(w.function_space().mesh().leaf_node(), "mesh")
    
        h5.write(w.leaf_node(), "w")
        
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
    
        with h5py.File(checkpoint_filepath, "r+") as h5:
            
            h5.create_dataset("t", data=time)
        
        
def read_checkpoint(checkpoint_filepath):
    """Read the checkpoint solution and time, perhaps to restart."""

    mesh = fenics.Mesh()
        
    with fenics.HDF5File(mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
    
        h5.read(mesh, "mesh", True)
    
    W_ele = make_mixed_fe(mesh.ufl_cell())

    W = fenics.FunctionSpace(mesh, W_ele)

    w = fenics.Function(W)

    with fenics.HDF5File(mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
    
        h5.read(w, "w")
        
    with h5py.File(checkpoint_filepath, "r") as h5:
            
        time = h5["t"].value
        
    return w, time
    

def steady(W, w, w_n, steady_relative_tolerance):
    """Check if solution has reached an approximately steady state."""
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(w_n, "L2")
    
    phaseflow.helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady
    
    
if __name__=="__main__":

    run()
