""" This module tests iterative solvers. """
from .context import phaseflow, tests
import fenics


def test_stefan_problem_benchmark_lu():

    tests.test_benchmarks.BenchmarkTest(phaseflow.benchmarks.StefanProblem(linear_solver = "lu")).run()


def test_stefan_problem_benchmark_gmres():

    tests.test_benchmarks.BenchmarkTest(phaseflow.benchmarks.StefanProblem(linear_solver = "gmres")).run()


def test_stefan_problem_benchmark_bicgstab():

    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.StefanProblem(linear_solver = "bicgstab")).run()


def test_stefan_problem_benchmark_cg():

    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.StefanProblem(linear_solver = "cg")).run()
    

def test_lid_driven_cavity_benchmark_lu():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "lu")).run()
    

def test_lid_driven_cavity_benchmark_gmres():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.LidDrivenCavity(mesh_size = 20, linear_solver = "gmres")).run()
        

def test_heat_driven_cavity_benchmark_lu():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.HeatDrivenCavity(mesh_size = 20, linear_solver = "lu")).run()
    

def test_heat_driven_cavity_benchmark_gmres():
    
    tests.test_benchmarks.BenchmarkTest(
        phaseflow.benchmarks.HeatDrivenCavity(mesh_size = 20, linear_solver = "gmres")).run()
    