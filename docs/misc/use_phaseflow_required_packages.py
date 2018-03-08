""" This module is a minimal example which uses the Python packages required by Phaseflow.

Run this as a script, i.e.

    $ python3 use_phaseflow_required_packages.py

The expected output is

    Solving linear variational problem.
    
If this runs without error, then Phaseflow should also work.

Note that while older versions of FEniCS and Python might work for this module,
Phaseflow requires fenics>=2017.1.0 and Python 3.

See the Phaseflow repository at https://github.com/geo-fluid-dynamics/phaseflow-fenics .

Alternatively, to test Phaseflow directly, do

    git clone git@github.com:geo-fluid-dynamics/phaseflow-fenics.git

    python3 -m pytest -k lid_driven_cavity phaseflow-fenics
"""
import fenics
import h5py


def boundary(x, on_boundary):
    """ Mark the boundary for applying Dirichlet boundary conditions. """
    return on_boundary
        
        
def solve_poisson_problem(mesh, boundary_values_expression, forcing_function):
    """ Solve the Poisson problem with P2 elements. """
    V = fenics.FunctionSpace(mesh, "P", 2)
    
    u, v = fenics.TrialFunction(V), fenics.TestFunction(V)
    
    solution = fenics.Function(V)
    
    fenics.solve(fenics.dot(fenics.grad(v), fenics.grad(u))*fenics.dx == v*forcing_function*fenics.dx,
        solution, 
        fenics.DirichletBC(V, boundary_values_expression, boundary))
    
    return solution


def checkpoint(solution, time, checkpoint_filepath):
    """ Write a checkpoint file (with solution and time). 
    
    The Poisson problem implemented in this module is not time dependent;
    but Phaseflow simulations are time dependent, and require h5py in addition to 
    the built-in `fenics.HDF5File` for checkpointing/restarting between time steps.
    """
    with fenics.HDF5File(fenics.mpi_comm_world(), checkpoint_filepath, "w") as h5:
        
        h5.write(solution.function_space().mesh(), "mesh")
    
        h5.write(solution, "solution")
        
    with h5py.File(checkpoint_filepath, "r+") as h5:
        
        h5.create_dataset("time", data = time)
    
    
if __name__=="__main__":
    """ Run the solver for a manufactured problem and call the checkpoint routine. """
    solution = solve_poisson_problem(mesh = fenics.UnitSquareMesh(8, 8),
        boundary_values_expression = fenics.Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree = 2),
        forcing_function = fenics.Constant(-6.0))
    
    checkpoint(solution = solution, time = 0., checkpoint_filepath = "checkpoint.h5")
    