import fenics
import phaseflow
        

T_f = 0.01

def melt_toy_pcm(output_dir = "output/melt_octadecane_pcm/",
        initial_values = None,
        start_time = 0.,
        end_time = 80.):
    
    T_hot = 1.
    
    T_cold = -0.1
    
    if initial_values is None:
    
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
    
    
        #
        initial_pci_position = 1./float(initial_mesh_size)/2.**(initial_hot_wall_refinement_cycles - 1)
        
        fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"),
                T_hot = T_hot, T_cold = T_cold, initial_pci_position = initial_pci_position,
                element = mixed_element),
            W)
            
    else:
    
        W = initial_values.function_space()
        
        
    # Run phaseflow.
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
        start_time = start_time,
        end_time = end_time,
        stop_when_steady = True,
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = 0.025,
        adaptive = True,
        adaptive_metric = "phase_only",
        adaptive_solver_tolerance = 1.e-5,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        nlp_relaxation = 1.,
        initial_values = initial_values,
        boundary_conditions = [
            fenics.DirichletBC(W.sub(0), (0., 0.), walls),
            fenics.DirichletBC(W.sub(2), T_hot, hot_wall),
            fenics.DirichletBC(W.sub(2), T_cold, cold_wall)],
        output_dir = output_dir)
    
    return w
    
    
def melt_octadecane_pcm():

    w = melt_toy_pcm(end_time = 20.)
    
    w, time = phaseflow.read_checkpoint("output/melt_octadecane_pcm/checkpoint_t20.0.h5")
    
    melt_toy_pcm(initial_values = w,
        output_dir = "output/melt_octadecane_pcm/restart_t20/",
        start_time = time, end_time = 80.)

    
    
if __name__=='__main__':

    melt_octadecane_pcm()
    