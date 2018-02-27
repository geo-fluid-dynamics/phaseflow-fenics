""" This module runs the unit test suite. """
from .context import phaseflow
import fenics
import tempfile


def test_1d_output_unit__ci__():

    sim = phaseflow.octadecane.Simulation()
    
    sim.mesh = fenics.UnitIntervalMesh(5)
    
    sim.update_element()
    
    function_space = fenics.FunctionSpace(sim.mesh, sim.element)
    
    state = phaseflow.state.State(function_space, sim.element)
    
    with phaseflow.helpers.SolutionFile(tempfile.mkdtemp() + "/output/test_1D_output/solution.xdmf") \
            as solution_file:
     
        state.write_solution(solution_file)
    
        
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


def test_time_dependent_unit():

    benchmark = phaseflow.octadecane_benchmarks.HeatDrivenCavityBenchmarkSimulation()
    
    benchmark.end_time = 0. + 2.*benchmark.timestep_size
    
    benchmark.run(verify = False)
    
    assert(not (benchmark.state.time == benchmark.old_state.time))
    
    assert(not all(benchmark.state.solution.vector() == benchmark.old_state.solution.vector()))

    
def test_checkpoint_and_restart__ci__():

    benchmark = phaseflow.octadecane_benchmarks.LidDrivenCavityBenchmarkSimulation()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    benchmark2 = phaseflow.octadecane_benchmarks.LidDrivenCavityBenchmarkSimulation()
    
    benchmark2.read_checkpoint(
        benchmark.output_dir + "/checkpoint_t" + str(benchmark.end_time) + ".h5")
    
    assert(benchmark.model.state.time == benchmark2.model.state.time)
    
    solution = benchmark.state.solution.leaf_node()
    
    solution2 = benchmark2.state.solution.leaf_node()
    
    assert(fenics.errornorm(solution, solution2) < 1.e-15)
    
    assert(all(solution.vector() == solution2.vector()))
    
    benchmark2.end_time *= 2.
    
    benchmark2.prefix_output_dir_with_tempdir = True
    
    benchmark2.run()
