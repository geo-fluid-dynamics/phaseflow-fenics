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
    velocity_degree = pressure_degree + 1
    
    velocity_element = fenics.VectorElement("P", cell, velocity_degree)
    
    pressure_element = fenics.FiniteElement("P", cell, pressure_degree)

    temperature_element = fenics.FiniteElement("P", cell, temperature_degree)

    mixed_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])
    
    return mixed_element

    
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
    
    velocity, pressure, temperature = solution.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for i, var in enumerate([velocity, pressure, temperature]):
    
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
  
  
def run(output_dir = "output/wang2010_natural_convection_air",
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        stefan_number = 0.045,
        liquid_viscosity = 1.,
        solid_viscosity = 1.e8,
        gravity = (0., -1.),
        m_B = None,
        ddT_m_B = None,
        penalty_parameter = 1.e-7,
        temperature_of_fusion = -1.e12,
        regularization_smoothing_factor = 0.005,
        initial_values = [],
        boundary_conditions = [],
        start_time = 0.,
        end_time = 10.,
        time_step_size = 1.e-3,
        stop_when_steady = True,
        steady_relative_tolerance=1.e-4,
        adaptive = False,
        adaptive_metric = "all",
        adaptive_solver_tolerance = 1.e-4,
        nlp_absolute_tolerance = 1.e-8,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        nlp_relaxation = 1.):
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
    
    
    # Report arguments.
    phaseflow.helpers.print_once("Running Phaseflow with the following arguments:")
    
    phaseflow.helpers.print_once(phaseflow.helpers.arguments())
    
    phaseflow.helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + "/arguments.txt", "w")
        
        arguments_file.write(str(phaseflow.helpers.arguments()))

        arguments_file.close()
    
    
    # Use function space and mesh from initial values.
    W = initial_values.function_space()
    
    mesh = W.mesh()
    
    
    # Check if 1D/2D/3D.
    dimensionality = mesh.type().dim()
    
    phaseflow.helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Set the initial values.
    w_n = initial_values
       
    u_n, p_n, T_n = fenics.split(w_n)
        
    
    # Set the variational form.
    """Set local names for math operators to improve readability."""
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    """The linear, bilinear, and trilinear forms b, a, and c, follow the common notation 
    for applying the finite element method to the incompressible Navier-Stokes equations,
    e.g. from danaila2014newton and huerta2003fefluids.
    """
    def b(u, q):
        return -div(u)*q  # Divergence
    
    
    def D(u):
    
        return sym(grad(u))  # Symmetric part of velocity gradient
    
    
    def a(mu, u, v):
        
        return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    
    def c(w, z, v):
        
        return dot(dot(grad(z), w), v)  # Convection of the velocity field
    
    
    Delta_t = fenics.Constant(time_step_size)
    
    Re = fenics.Constant(reynolds_number)
    
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    Ste = fenics.Constant(stefan_number)
    
    g = fenics.Constant(gravity)
    
    def f_B(T):
    
        return m_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g  # Buoyancy force, $f = ma$
    
    
    gamma = fenics.Constant(penalty_parameter)
    
    T_f = fenics.Constant(temperature_of_fusion)
    
    r = fenics.Constant(regularization_smoothing_factor)
    
    def P(T):
    
        return 0.5*(1. - fenics.tanh((T_f - T)/r))  # Regularized phase field.
    
    
    mu_l = fenics.Constant(liquid_viscosity)
    
    mu_s = fenics.Constant(solid_viscosity)
    
    def mu(T):
    
        return mu_s + (mu_l - mu_s)*P(T) # Variable viscosity.
    
    
    L = 1./Ste  # Latent heat
    
    v, q, phi = fenics.TestFunctions(W)
    
    w = fenics.Function(W)
    
    u, p, T = fenics.split(w)

    F = (
        b(u, q) - gamma*p*q
        + dot(u - u_n, v)/Delta_t
        + c(u, u, v) + b(v, p) + a(mu(T), u, v)
        + dot(f_B(T), v)
        + 1./Delta_t*(T - T_n)*phi
        - dot(T*u, grad(phi)) 
        + 1./Pr*dot(grad(T), grad(phi))
        + 1./Delta_t*L*(P(T) - P(T_n))*phi
        )*fenics.dx

    def ddT_f_B(T):
        
        return ddT_m_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g
    
    
    def sech(theta):
    
        return 1./fenics.cosh(theta)
    
    
    def dP(T):
    
        return sech((T_f - T)/r)**2/(2.*r)

        
    def dmu(T):
    
        return (mu_l - mu_s)*dP(T)
    
    
    # Set the Jacobian (formally the Gateaux derivative) in variational form.
    delta_w = fenics.TrialFunction(W)
    
    delta_u, delta_p, delta_T = fenics.split(delta_w)
    
    w_k = w
    
    u_k, p_k, T_k = fenics.split(w_k)
    
    JF = (
        b(delta_u, q) - gamma*delta_p*q 
        + dot(delta_u, v)/Delta_t
        + c(u_k, delta_u, v) + c(delta_u, u_k, v) + b(v, delta_p)
        + a(delta_T*dmu(T_k), u_k, v) + a(mu(T_k), delta_u, v) 
        + dot(delta_T*ddT_f_B(T_k), v)
        + 1./Delta_t*delta_T*phi
        - dot(T_k*delta_u, grad(phi))
        - dot(delta_T*u_k, grad(phi))
        + 1./Pr*dot(grad(delta_T), grad(phi))
        + 1./Delta_t*L*delta_T*dP(T_k)*phi
        )*fenics.dx

        
    # Set the functional metric for the error estimator for adaptive mesh refinement.
    """I haven't found a good way to make this flexible yet.
    Ideally the user would be able to write the metric, but this would require giving the user
    access to much data that phaseflow is currently hiding.
    """
    M = P(T_k)*fenics.dx
    
    if adaptive_metric == "phase_only":
    
        pass
        
    elif adaptive_metric == "all":
        
        M += T_k*fenics.dx
        
        for i in range(dimensionality):
        
            M += u_k[i]*fenics.dx
            
    else:
        
        assert(False)
        
    # Make the problem.
    problem = fenics.NonlinearVariationalProblem(F, w_k, boundary_conditions, JF)
    
    
    # Make the solver.
    """ For the purposes of this project, it would be better to just always use the adaptive solver; but
    unfortunately the adaptive solver encounters nan's whenever evaluating the error for problems not 
    involving phase-change. So far my attempts at writing a MWE to reproduce the  issue have failed.
    """   
    if adaptive:
    
        solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
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
    
    
    # Open a context manager for the output file.
    solution_filepath = output_dir + "/solution.xdmf"
    
    with fenics.XDMFFile(solution_filepath) as solution_file:

        time = start_time 
        
        
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
            
            write_solution(solution_file, w_k, time, solution_filepath)
            
            
            # Write checkpoint files.
            write_checkpoint(checkpoint_filepath = output_dir + "/checkpoint_t" + str(time) + ".h5",
                w = w_k,
                time = time)
            
            
            # Check for steady state.
            if stop_when_steady and steady(W, w_k, w_n, steady_relative_tolerance):
            
                phaseflow.helpers.print_once("Reached steady state at time t = " + str(time))
                
                break
                
                
            # Set initial values for next time step.
            w_n.leaf_node().vector()[:] = w_k.leaf_node().vector()
            
            
            # Report progress.
            progress.update(time / end_time)
            
            if time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                phaseflow.helpers.print_once("Reached end time, t = " + str(end_time))
            
                break
    
    
    # Return the interpolant to sample inside of Python.
    w_k.rename("w", "state")
    
    return w_k, time
    
    
if __name__=="__main__":

    run()
