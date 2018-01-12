from builtins import str
from builtins import range
from .context import phaseflow
import fenics
import scipy.optimize as opt

T_r = 0.1

def melt_toy_pcm(output_dir = "output/test_melt_toy_pcm/"):
    
    
    # Make the mesh.
    initial_mesh_size = 1
    
    mesh = fenics.UnitSquareMesh(initial_mesh_size, initial_mesh_size, 'crossed')
    
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
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    
    # Set the semi-phase-field mapping
    r = 0.025
    
    def phi(T):

        return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
    def sech(theta):
    
        return 1./fenics.cosh(theta)
    
    
    def dphi(T):
    
        return -sech((T_r - T)/r)**2/(2.*r)
    
    
    # Run phaseflow.
    T_hot = 1.
    
    T_cold = -0.1
    
    initial_pci_position = 0.001
    
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    solution = fenics.Function(function_space)
    
    p, u, T = fenics.split(solution)
    
    phaseflow.run(solution = solution,
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"),
                T_hot = T_hot, T_cold = T_cold, initial_pci_position = initial_pci_position,
                element = mixed_element),
            function_space),
        boundary_conditions = [
            fenics.DirichletBC(function_space.sub(1), (0., 0.), walls),
            fenics.DirichletBC(function_space.sub(2), T_hot, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_cold, cold_wall)],
        stefan_number = 1.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        time_step_size = 1.e-3,
        end_time = 0.02,
        stop_when_steady = True,
        semi_phasefield_mapping = phi,
        semi_phasefield_mapping_derivative = dphi,
        adaptive_goal_functional = phi(T)*fenics.dx,
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
        output_dir = output_dir)
    
    return solution
    
    
def test_melt_toy_pcm__regression():

    solution = melt_toy_pcm()
    
    
    # Verify against a reference solution.
    pci_y_position_to_check =  0.875
    
    reference_pci_x_position = 0.226
    
    def T_minus_T_r(x):
    
        values = solution.leaf_node()(fenics.Point(x, pci_y_position_to_check))
        
        return values[3] - T_r

        
    pci_x_position = opt.newton(T_minus_T_r, 0.01)
    
    assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
    
    
if __name__=='__main__':

    test_melt_toy_pcm__regression()
    