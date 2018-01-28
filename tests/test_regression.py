""" This module runs the regression test suite. """
from .context import phaseflow
import scipy.optimize
import fenics


class AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression(
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingOctadecanePCM):

    def __init__(self, 
            depth_3d = None, 
            initial_mesh_size = (1, 1),
            initial_hot_wall_refinement_cycles = 6,
            end_time = 30.,
            quadrature_degree = 8,
            adaptive_solver_tolerance = 1.e-5):
    
        phaseflow.benchmarks.AdaptiveConvectionCoupledMeltingOctadecanePCM.__init__(self, 
            timestep_size = 10., 
            end_time = end_time, 
            quadrature_degree = quadrature_degree,
            depth_3d = depth_3d, 
            initial_mesh_size = initial_mesh_size, 
            initial_hot_wall_refinement_cycles = initial_hot_wall_refinement_cycles,
            adaptive_solver_tolerance = adaptive_solver_tolerance)
    
        self.output_dir += "regression/"
        
        
    def verify(self):
        """ Test regression based on a previous solution from Phaseflow.
        In Paraview, the $T = 0.01$ (i.e. the regularization_central_temperature) contour was drawn
        at time $t = 30.$ (i.e. the end_time).
        A point from this contour in the upper portion of the domain, 
        where the PCI has advanced more quickly, was recorded to be (0.278, 0.875).
        This was checked for commit a8a8a039e5b89d71b6cceaef519dfbf136322639.
        """
        pci_y_position_to_check =  0.88
        
        reference_pci_x_position = 0.28
        
        def T_minus_T_r(x):
            
            values = self.model.state.solution.leaf_node()(fenics.Point(x, pci_y_position_to_check))
            
            return values[3] - self.regularization_central_temperature

            
        pci_x_position = scipy.optimize.newton(T_minus_T_r, 0.01)
        
        assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
        
        
class AdaptiveConvectionCoupledMeltingOctadecanePCM_3D_Regression(
        AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression):

    def __init__(self):
    
        AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression.__init__(self, 
            end_time = 10., 
            quadrature_degree = 7,
            depth_3d = 0.5, 
            initial_mesh_size = (1, 1, 1), 
            initial_hot_wall_refinement_cycles = 4,
            adaptive_solver_tolerance = 5.e-4)
    
        self.output_dir += "3d/"
        
        
    def verify(self):

        pci_y_position_to_check =  0.88
        
        reference_pci_x_position = 0.19
        
        def T_minus_T_r(x):
            
            values = self.model.state.solution.leaf_node()(fenics.Point(x, pci_y_position_to_check, 0.))
            
            return values[4] - self.regularization_central_temperature

            
        pci_x_position = scipy.optimize.newton(T_minus_T_r, 0.01)
        
        assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
        

def test_adaptive_convection_coupled_melting_octadecane_pcm_regression():

    AdaptiveConvectionCoupledMeltingOctadecanePCM_Regression().run()


def test_adaptive_convection_coupled_melting_octadecane_pcm_threed_regression():

    AdaptiveConvectionCoupledMeltingOctadecanePCM_3D_Regression().run()    
