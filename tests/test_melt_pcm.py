from builtins import str
from builtins import range
from .context import phaseflow
import fenics
import scipy.optimize as opt

T_f = 0.01

def melt_toy_pcm(output_dir = "output/test_melt_toy_pcm/"):
    
    
    # Make the mesh.
    initial_mesh_size = 1
    
    mesh = fenics.UnitSquareMesh(initial_mesh_size, initial_mesh_size)
    
    initial_hot_wall_refinement_cycles = 6
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        hot_wall.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()
    
    
    #
    mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
    W = fenics.FunctionSpace(mesh, mixed_element)
    
    
    # Run phaseflow.
    T_hot = 1.
    
    T_cold = -0.1
    
    initial_pci_position = 1./float(initial_mesh_size)/2.**(initial_hot_wall_refinement_cycles - 1)
    
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    w, mesh = phaseflow.run(
        stefan_number = 0.045,
        rayleigh_number = 3.27e5,
        prandtl_number = 56.2,
        solid_viscosity = 1.e8,
        liquid_viscosity = 1.,
        time_step_size = 1.,
        end_time = 2.,
        stop_when_steady = True,
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = 0.05,
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 100,
        nlp_relaxation = 1.,
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"),
                T_hot = T_hot, T_cold = T_cold, initial_pci_position = initial_pci_position,
                element = mixed_element),
            W),
        boundary_conditions = [
            fenics.DirichletBC(W.sub(0), (0., 0.), walls),
            fenics.DirichletBC(W.sub(2), T_hot, hot_wall),
            fenics.DirichletBC(W.sub(2), T_cold, cold_wall)],
        output_dir = output_dir)
    
    return w
    
    
def test_melt_toy_pcm__regression():

    w = melt_toy_pcm()
    
    
    # Verify against a reference solution.
    pci_y_position_to_check =  0.875
    
    reference_pci_x_position = 0.226
    
    def T_minus_T_f(x):
    
        wval = w.leaf_node()(fenics.Point(x, pci_y_position_to_check))
        
        return wval[3] - T_f

        
    pci_x_position = opt.newton(T_minus_T_f, 0.01)
    
    assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
    
    
if __name__=='__main__':

    test_melt_toy_pcm__regression()
    