import fenics
import phaseflow


T_f = 0.1

def melt_toy_pcm(output_dir = "output/melt_toy_pcm/"):
    
    
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
        
    W = fenics.FunctionSpace(mesh, mixed_element)
    
    
    # Run phaseflow.
    T_hot = 1.
    
    T_cold = -0.1
    
    initial_pci_position = 0.001
    
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    w, mesh = phaseflow.run(
        stefan_number = 1.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        time_step_size = 1.e-3,
        end_time = 0.05,
        stop_when_steady = True,
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = 0.025,
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
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

    
if __name__=='__main__':

    melt_toy_pcm()
    