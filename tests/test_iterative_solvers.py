""" This module tests iterative solvers. """
from .context import phaseflow, tests
import fenics


def test_stefan_problem_benchmark_lu():

    tests.test_benchmarks.BenchmarkTest(phaseflow.benchmarks.StefanProblem(linear_solver = "lu")).run()


def test_stefan_problem_benchmark_cg():

    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.StefanProblem(linear_solver = "cg")).run()
    

def test_lid_driven_cavity_benchmark_lu():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "lu")).run()


def test_lid_driven_cavity_benchmark_cg():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "cg")).run()


def test_lid_driven_cavity_benchmark_bicgstab():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "bicgstab")).run()

        
def test_lid_driven_cavity_benchmark_print_linear_system():
    
    test = tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 3, linear_solver = "gmres"))
        
    test.benchmark.setup_solver()
        
    test.benchmark.solver.fenics_solver.parameters["print_matrix"] = True
    
    test.benchmark.solver.fenics_solver.parameters["print_rhs"] = True
    
    test.benchmark.solver.fenics_solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = 100
    
    test.run()
    

def test_lid_driven_cavity_benchmark_gmres():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "gmres")).run()
        

def test_heat_driven_cavity_benchmark_lu():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.HeatDrivenCavity(mesh_size = 20, linear_solver = "lu")).run()
    

def test_heat_driven_cavity_benchmark_gmres():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.HeatDrivenCavity(mesh_size = 20, linear_solver = "gmres")).run()
    
    
def test_adaptive_convection_coupled_melting_octadecane_pcm_regression_print_linear_system():

    test = tests.test_regression.AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression(linear_solver = "gmres")
    
    test.setup_solver()
        
    test.solver.fenics_solver.parameters["nonlinear_variational_solver"]["print_matrix"] = True
    
    test.solver.fenics_solver.parameters["nonlinear_variational_solver"]["print_rhs"] = True
    
    test.solver.fenics_solver.parameters["nonlinear_variational_solver"]["newton_solver"]["krylov_solver"]\
        ["maximum_iterations"] = 10
    
    test.run()
