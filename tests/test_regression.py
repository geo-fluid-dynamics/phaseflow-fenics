""" This module runs the regression test suite. """
from .context import phaseflow
import scipy.optimize


class AdaptiveConvectionCoupledMeltingToyPCM(
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingOctadecanePCM):

    def __init__(self):
    
        self.regularization_central_temperature = 0.1
        
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingPCM.__init__(self, 
            initial_mesh_size = 1, 
            initial_hot_wall_refinement_cycles = 6,
            initial_pci_position = 0.001,
            T_hot = 1.,
            T_cold = -0.1,
            stefan_number = 1.,
            rayleigh_number = 1.e6,
            prandtl_number = 0.71,
            solid_viscosity = 1.e4,
            liquid_viscosity = 1.,
            timestep_size = 1.e-3,
            regularization_central_temperature = self.regularization_central_temperature,
            regularization_smoothing_parameter = 0.025,
            end_time = 0.02)
    
        self.adaptive_solver_tolerance = 1.e-4
        
        self.nlp_absolute_tolerance = 1.e-8
        
        self.nlp_relative_tolerance = 1.e-8
        
        self.stop_when_steady = False
        
        self.output_dir = "output/benchmarks/adaptive_convection_coupled_melting_toy_pcm_regression/"
        
        
    def verify(self):
        """ Test regression based on a previous solution from Phaseflow. """
        pci_y_position_to_check =  0.875
        
        reference_pci_x_position = 0.226
        
        def T_minus_T_r(x):
        
            values = self.model.state.solution.leaf_node()(fenics.Point(x, pci_y_position_to_check))
            
            return values[3] - self.regularization_central_temperature

            
        pci_x_position = scipy.optimize.newton(T_minus_T_r, 0.01)
        
        assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
        


def test__failing__adaptive_convection_coupled_melting_toy_pcm_regression():

    AdaptiveConvectionCoupledMeltingToyPCM().run()
    
    
if __name__=='__main__':
    
    test__failing__adaptive_convection_coupled_melting_toy_pcm_regression()
        