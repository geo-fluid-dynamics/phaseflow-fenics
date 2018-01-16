import fenics
import phaseflow


def melt_octadecane_pcm(output_dir = "output/melt_octadecane_pcm/",
        initial_values = None,
        time = 0.,
        end_time = 80.,
        time_step_size = 1.,
        adaptive_solver_tolerance = 1.e-5,
        solid_viscosity = 1.e8,
        nlp_max_iterations = 100):
    
    T_hot = 1.
    
    T_cold = -0.01
    
    if initial_values is None:
    
        # Make the mesh.
        """Begin with the coarsest possible unit square, then refine near the hot wall.
        Insufficient initial refinement will cause the Newton method to fail."""
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
        
        
        # Set the initial values.
        mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
            
        function_space = fenics.FunctionSpace(mesh, mixed_element)

        initial_pci_position = 1./float(initial_mesh_size)/2.**(initial_hot_wall_refinement_cycles - 1)
        
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "(T_hot - T_cold)*(x[0] < initial_pci_position) + T_cold"),
                T_hot = T_hot, T_cold = T_cold, initial_pci_position = initial_pci_position,
                element = mixed_element),
            function_space)
            
    else:
    
        function_space = initial_values.function_space()
        
        
    # Set the semi-phase-field mapping
    """ The Newton method's behaviour is very sensitive to the value of the smoothing parameter $r$."""
    r = 0.025
    
    T_r = 0.01
    
    def phi(T):

        return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
    def sech(theta):
    
        return 1./fenics.cosh(theta)
    
    
    def dphi(T):
    
        return -sech((T_r - T)/r)**2/(2.*r)
        
        
    # Specify approximate quadrature degree.
    """By default, FEniCS uses the exact quadrature rule.
    For this example, FEniCS warns that this yields a large number of quadrature points.
    Using too low of a degree causes the Newton method to fail,
    but we can indeed succeed with less-than-exact quadrature."""
    quadrature_degree = 8
    
    dx = fenics.dx(metadata={'quadrature_degree': quadrature_degree})

    
    # Set the boundary conditions.
    walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"
    
    hot_wall = "near(x[0],  0.)"
    
    cold_wall = "near(x[0],  1.)"
    
    boundary_conditions = [
        fenics.DirichletBC(function_space.sub(1), (0., 0.), walls),
        fenics.DirichletBC(function_space.sub(2), T_hot, hot_wall),
        fenics.DirichletBC(function_space.sub(2), T_cold, cold_wall)]
    
    
    # Initialize the solution.
    """The values handed to Phaseflow will be used as the Newton method's initial guess."""
    solution = fenics.Function(function_space)
    
    solution.leaf_node().vector()[:] = initial_values.leaf_node().vector()
    
    
    # Set the adaptive goal.
    p, u, T = fenics.split(solution)
    
    adaptive_goal_functional = phi(T)*dx
    
    
    # Run Phaseflow.
    phaseflow.run(solution = solution,
        initial_values = initial_values,
        boundary_conditions = boundary_conditions,
        time = time,
        stefan_number = 0.045,
        rayleigh_number = 3.27e5,
        prandtl_number = 56.2,
        solid_viscosity = solid_viscosity,
        liquid_viscosity = 1.,
        time_step_size = time_step_size,
        end_time = end_time,
        semi_phasefield_mapping = phi,
        semi_phasefield_mapping_derivative = dphi,
        adaptive_goal_functional = adaptive_goal_functional,
        adaptive_solver_tolerance = adaptive_solver_tolerance,
        nlp_max_iterations = nlp_max_iterations,
        quadrature_degree = quadrature_degree,
        output_dir = output_dir)

    return solution, time
    
    
def run_melt_octadecane_pcm_to_t80():

    melt_octadecane_pcm(output_dir = "output/melt_octadecane_pcm/",
        time = 0., 
        adaptive_solver_tolerance = 1.e-5,
        end_time = 36.)
    
    solution, time = phaseflow.read_checkpoint("output/melt_octadecane_pcm/checkpoint_t36.0.h5")
    
    assert(abs(time - 36.) < 1.e-8)
    
    melt_toy_pcm(initial_values = solution,
        output_dir = "output/melt_octadecane_pcm/restart_t36/",
        time = time, 
        end_time = 80.,
        adaptive_solver_tolerance = 0.5e-5)
    
    
if __name__=='__main__':

    run_melt_octadecane_pcm_to_t80()
    