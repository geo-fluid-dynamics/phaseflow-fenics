""" This module runs the regression test suite. """
from .context import phaseflow
import scipy.optimize
import fenics


class AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression(
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingOctadecanePCM):

    def __init__(self):
    
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingOctadecanePCM.__init__(self, 
            solid_viscosity = 1.e4,
            stefan_number = 0.045,
            regularization_smoothing_parameter = 0.025,
            timestep_size = 10.,
            end_time = 30.,
            adaptive_solver_tolerance = 1.e-5,
            nlp_relative_tolerance = 1.e-8)
    
        
        
        self.output_dir = "output/benchmarks/adaptive_convection_coupled_melting_octadecane_pcm_regression/"
        
        
    def verify(self):
        """ Test regression based on a previous solution from Phaseflow. """
        pci_y_position_to_check =  0.875
        
        reference_pci_x_position = 0.226
        
        def T_minus_T_r(x):
        
            values = self.model.state.solution.leaf_node()(fenics.Point(x, pci_y_position_to_check))
            
            return values[3] - self.regularization_central_temperature

            
        pci_x_position = scipy.optimize.newton(T_minus_T_r, 0.01)
        
        assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
        


def test__failing__adaptive_convection_coupled_melting_octadecane_pcm_regression():

    AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression().run()
    
    
if __name__=='__main__':
    
    test__failing__adaptive_convection_coupled_melting_octadecane_pcm_regression()
        