from builtins import str
from builtins import range
from .context import phaseflow
import fenics


def verify_against_ghia1982(w, mesh):

    data = {'Re': 100, 'x': 0.5,
        'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 
              0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000],
        'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, 
               -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
            
            ux = wval[0]
            
            assert(abs(ux - true_ux) < 2.e-2)
            

def test_variable_viscosity__nightly():

    lid = "near(x[1],  1.)"

    ymin = -0.25
    
    fixed_walls = "near(x[0],  0.) | near(x[0],  1.) | near(x[1],  " + str(ymin) + ")"

    left_middle = "near(x[0], 0.) && near(x[1], 0.5)"
    
    output_dir = "output/test_variable_viscosity"
    
    mesh = fenics.RectangleMesh(fenics.Point(0., ymin), fenics.Point(1., 1.), 20, 25, "crossed")
    
    
    # Refine the initial PCI.
    initial_pci_refinement_cycles = 4
    
    class PCI(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return fenics.near(x[1], 0.)

            
    pci = PCI()
    
    for i in range(initial_pci_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        pci.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()
    
    
    # Run the simulation.
    w, mesh = phaseflow.run(
        mesh = mesh,
        end_time = 20.,
        time_step_size = 1.,
        stop_when_steady = True,
        steady_relative_tolerance = 1.e-4,
        thermal_conductivity = 0.,
        liquid_viscosity = 0.01,
        solid_viscosity = 1.e6,
        temperature_of_fusion = -0.01,
        regularization_smoothing_factor = 0.01,
        gravity = (0., 0.),
        stefan_number = 1.e16,
        adaptive = False,
        output_dir = output_dir,
        initial_values_expression = (lid, "0.", "0.", "1. - 2.*(x[1] <= 0.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("1.", "0."), 'degree': 3, 'location_expression': lid, 'method': 'topological'},
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': fixed_walls, 'method': 'topological'}])
    
    
    # Verify against the known solution.
    verify_against_ghia1982(w, mesh)

    
if __name__=='__main__':

    test_variable_viscosity__nightly()
    