""" This module runs the unit test suite. """
from .context import phaseflow
import fenics
import tempfile


def test_1d_output_unit__ci__():

    mesh = fenics.UnitIntervalMesh(5)
    
    mixed_element = phaseflow.pure.make_mixed_element(mesh.ufl_cell())
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    state = phaseflow.core.State(function_space, mixed_element)
    
    with phaseflow.core.SolutionFile(tempfile.mkdtemp() + "/output/test_1D_output/solution.xdmf") \
            as solution_file:
     
        state.write_solution_to_xdmf(solution_file)
    
        
def test_1d_velocity_unit__ci__():

    mesh = fenics.UnitIntervalMesh(5)

    V = fenics.VectorFunctionSpace(mesh, "P", 1)

    u = fenics.Function(V)

    bc = fenics.DirichletBC(V, [10.0], "x[0] < 0.5")

    print(bc.get_boundary_values())
    
    
def test_xdmf_unit__ci__():

    solution_file = fenics.XDMFFile("test.xdmf")


'''This test seems to fail with fenics-2016.2.0. 
I vaguely recall seeing an issue on their Bitbucket which mentions having
not always used the proper context manager style with some of their file classes.'''
def test_xdmf_context_unit__ci__():

    with fenics.XDMFFile("test.xdmf") as solution_file:

        return


def test_checkpoint_and_restart__ci__():

    benchmark = phaseflow.benchmarks.LidDrivenCavity()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    benchmark2 = phaseflow.benchmarks.LidDrivenCavity()
    
    benchmark2.model.read_checkpoint(
        benchmark.output_dir + "/checkpoint_t" + str(benchmark.end_time) + ".h5")
    
    assert(benchmark.model.state.time == benchmark2.model.state.time)
    
    solution = benchmark.model.state.solution.leaf_node()
    
    solution2 = benchmark2.model.state.solution.leaf_node()
    
    assert(fenics.errornorm(solution, solution2) < 1.e-15)
    
    assert(all(solution.vector() == solution2.vector()))
    
    benchmark2.end_time *= 2.
    
    benchmark2.prefix_output_dir_with_tempdir = True
    
    benchmark2.run()

    
def test_checkpoint_and_restart_adaptive__ci__():

    benchmark = phaseflow.benchmarks.AdaptiveLidDrivenCavity()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    benchmark2 = phaseflow.benchmarks.AdaptiveLidDrivenCavity()
    
    benchmark2.model.read_checkpoint(
        benchmark.output_dir + "/checkpoint_t" + str(benchmark.end_time) + ".h5")
    
    assert(benchmark.model.state.time == benchmark2.model.state.time)
    
    solution = benchmark.model.state.solution.leaf_node()
    
    solution2 = benchmark2.model.state.solution.leaf_node()
    
    assert(fenics.errornorm(solution, solution2) < 1.e-15)
    
    assert(all(solution.vector() == solution2.vector()))
    
    benchmark2.end_time *= 2.
    
    benchmark2.prefix_output_dir_with_tempdir = True
    
    benchmark2.run()
    