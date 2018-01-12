from .context import phaseflow
import fenics


def verify_against_ghia1982(solution):

    data = {'Re': 100, 'x': 0.5, 
        'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
            0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 
        'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, 
            -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
    
    bbt = solution.function_space().mesh().bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            values = solution(p)
            
            ux = values[1]
            
            assert(abs(ux - true_ux) < 2.e-2)


def test_lid_driven_cavity():

    lid = "near(x[1],  1.)"

    fixed_walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)"

    bottom_left_corner = "near(x[0], 0.) && near(x[1], 0.)"

    grid_size = 20
    
    mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), grid_size, grid_size)
        
    mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
    function_space = fenics.FunctionSpace(mesh, mixed_element)
    
    solution = fenics.Function(function_space)
    
    time = phaseflow.run(solution = solution,
        initial_values = fenics.interpolate(
            fenics.Expression(("0.", lid, "0.", "1."), element = mixed_element), function_space),
        boundary_conditions = [
            fenics.DirichletBC(function_space.sub(1), (1., 0.), lid),
            fenics.DirichletBC(function_space.sub(1), (0., 0.), fixed_walls)],
        end_time = 1.e12,
        time_step_size = 1.e12,
        liquid_viscosity = 0.01,
        gravity = (0., 0.),
        output_dir='output/test_lid_driven_cavity')

    verify_against_ghia1982(solution)

    
if __name__=='__main__':

    test_lid_driven_cavity()

