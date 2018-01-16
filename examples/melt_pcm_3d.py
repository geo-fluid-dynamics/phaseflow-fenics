import fenics
import phaseflow
        
def melt_pcm_3d(
        initial_mesh_size = [1, 1, 1],
        dt = 1.e-3,
        regularization_smoothing_factor = 0.025,
        initial_hot_wall_refinement_cycles = 4,
        output_dir="output/melt_pcm_3d/",
        end_time=0.02,
        nlp_max_iterations = 200,
        nlp_relaxation = 0.8,
        adaptive_solver_tolerance = 1.e-2):

    # Make the mesh.
    depth = 0.4
    
    mesh = fenics.BoxMesh(fenics.Point(0., 0., -depth/2.),
            fenics.Point(1., 1., depth/2.),
            initial_mesh_size[0], initial_mesh_size[1], initial_mesh_size[2])
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh, False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
                    break
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
        mesh = fenics.refine(mesh, cell_markers)
        
    
    # Set the semi-phase-field mapping
    r = regularization_smoothing_factor
    
    T_r = 0.1
    
    def phi(T):

        return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
    def sech(theta):
    
        return 1./fenics.cosh(theta)
    
    
    def dphi(T):
    
        return -sech((T_r - T)/r)**2/(2.*r)
        
        
    # Set initial values and initialize the solution.
    initial_pci_position = 1./float(initial_mesh_size[0])/2.**(initial_hot_wall_refinement_cycles - 1)
    
    T_hot = 1.
    
    T_cold = -0.1
    
    mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
            
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    initial_values = fenics.interpolate(
        fenics.Expression(
            ("0.", "0.", "0.", "0.", "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"),
            T_hot = T_hot, T_cold = T_cold, initial_pci_position = initial_pci_position,
            element = mixed_element),
        function_space)
            
    solution = fenics.Function(function_space)
    
    solution.leaf_node().vector()[:] = initial_values.leaf_node().vector()
    
    
    # Run Phaseflow.
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.) " \
        + " | near(x[2],  " + str(-depth/2.) + ") | near(x[2],  " + str(depth/2.) + ")"
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    p, u, T = fenics.split(solution)
    
    time = 0.
    
    phaseflow.run(solution,
        time = time,
        initial_values = initial_values,
        boundary_conditions = [
            fenics.DirichletBC(function_space.sub(1), (0., 0., 0.), walls),
            fenics.DirichletBC(function_space.sub(2), T_hot, hot_wall),
            fenics.DirichletBC(function_space.sub(2), T_cold, cold_wall)],
        stefan_number = 1.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        gravity = (0., -1., 0.),
        time_step_size = dt,
        end_time = end_time,
        semi_phasefield_mapping = phi,
        semi_phasefield_mapping_derivative = dphi,
        adaptive_goal_functional = phi(T)*fenics.dx,
        adaptive_solver_tolerance = adaptive_solver_tolerance,
        nlp_max_iterations = nlp_max_iterations,
        nlp_relaxation = nlp_relaxation,
        output_dir = output_dir)

    return solution, time
    
    
def run_melt_pcm_3d():
    
    solution, time = melt_pcm_3d(output_dir = "output/melt_pcm_3d/", 
        end_time = 0.02, 
        nlp_max_iterations = 200,
        nlp_relaxation = 0.8,
        dt = 1.e-3,
        regularization_smoothing_factor = 0.05)
        
    
if __name__=='__main__':

    run_melt_pcm_3d()
