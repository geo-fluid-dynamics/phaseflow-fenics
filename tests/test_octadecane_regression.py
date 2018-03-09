""" This module runs the regression test suite. """
from .context import phaseflow
import fenics


def test_convection_coupled_melting_octadecane_pcm_regression__ci__():

    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.CCMOctadecanePCMRegressionSimulation())
