from builtins import str
from builtins import range
from .context import phaseflow

import fenics

def verify_against_analytical(solution):
    """ Verify again analytical solution from Kai's MATLAB script."""
    analytical_solution = {'Ste': 0.045, 'time': 0.10, 'true_pci_pos': 0.094,
        'x': [0.00, 0.025, 0.050, 0.075, 0.10, 0.5, 1.], 
        'T': [1.0, 0.73, 0.47, 0.20, 0.00, -0.01, -0.01]}
    
    for x, true_T in zip(analytical_solution['x'], analytical_solution['T']):
    
        values = solution.leaf_node()(fenics.Point(x))
        
        T = values[2]
        
        assert(abs(T - true_T) < 1.e-2)
        
    
def refine_near_left_boundary(mesh, cycles):
    """ Refine mesh near the left boundary.
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    """
    for i in range(cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
                    break
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
                break # There should only be one such point in 1D.
                
        mesh = fenics.refine(mesh, cell_markers)
        
    return mesh
    
    
def stefan_problem(output_dir = "output/stefan_problem",
        stefan_number = 0.045,
        T_h = 1.,
        T_c = -0.01,
        regularization_smoothing_factor = 0.005,
        time_step_size = 0.001,
        end_time = 0.1,
        initial_uniform_cell_count = 1,
        initial_hot_wall_refinement_cycles = 10,
        adaptive = True):
    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    mesh = refine_near_left_boundary(mesh, initial_hot_wall_refinement_cycles)
    
    mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    initial_pci_position = 1./float(initial_uniform_cell_count)/2.**(initial_hot_wall_refinement_cycles - 1)
    
    T_r = 0.
        
    r = regularization_smoothing_factor
    
    def phi(T):

        return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
    def sech(theta):
    
        return 1./fenics.cosh(theta)
    
    
    def dphi(T):
    
        return -sech((T_r - T)/r)**2/(2.*r)
    
    
    # Set initial values.
    solution = fenics.Function(function_space)
    
    initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "(T_h - T_c)*(x[0] < initial_pci_position) + T_c"),
                T_h = T_h, T_c = T_c, initial_pci_position = initial_pci_position,
                element=mixed_element),
            function_space)
    
    
    #
    phaseflow.run(solution = solution,
        initial_values = initial_values,
        boundary_conditions = [
            fenics.DirichletBC(function_space.sub(2), T_h, "near(x[0],  0.)"),
            fenics.DirichletBC(function_space.sub(2), T_c, "near(x[0],  1.)")],
        prandtl_number = 1.,
        stefan_number = stefan_number,
        gravity = [0.],
        semi_phasefield_mapping = phi,
        semi_phasefield_mapping_derivative = dphi,
        adaptive = adaptive,
        adaptive_metric = "phase_only",
        adaptive_solver_tolerance = 1.e-6,
        end_time = end_time,
        time_step_size = time_step_size,
        output_dir = output_dir,)
        
    return solution
        
        
def test_stefan_problem_melt_Ste0p045_uniform_grid():
    
    solution = stefan_problem(output_dir = "output/test_stefan_problem_melt_Ste0p045_uniform_grid/",
        initial_uniform_cell_count = 311, initial_hot_wall_refinement_cycles = 0,
        adaptive = False)
    
    verify_against_analytical(solution)
    
    
def test_stefan_problem_melt_Ste0p045_amr():
    
    solution = stefan_problem(output_dir = "output/test_stefan_problem_melt_Ste0p045_amr/",
        initial_uniform_cell_count = 4, initial_hot_wall_refinement_cycles = 8,
        adaptive = True)
    
    verify_against_analytical(solution)

    
if __name__=='__main__':
    
    test_stefan_problem_melt_Ste0p045_uniform_grid()
    
    test_stefan_problem_melt_Ste0p045_adaptive()
    