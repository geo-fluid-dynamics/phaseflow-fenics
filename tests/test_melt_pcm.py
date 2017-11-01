import fenics
import phaseflow
import scipy.optimize as opt


def test_melt_toy_pcm():
    
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
    T_hot = 1.
    
    T_cold = -0.1

    T_f = 0.1
    
    w, mesh = phaseflow.run(
        stefan_number = 1.,
        rayleigh_number = 1.,
        prandtl_number = 1.,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        mesh = mesh,
        time_step_size = 1.e-3,
        end_time = 0.02,
        stop_when_steady = True,
        regularization = {'T_f': T_f, 'r': 0.05},
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "("+str(T_hot)+" - "+str(T_cold)+")*(x[0] < 0.001) + "+str(T_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(T_hot), 'degree': 2,
                'location_expression': "near(x[0],  0.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(T_cold), 'degree': 2,
                'location_expression': "near(x[0],  1.)",
                'method': "topological"}],
        output_dir = 'output/test_melt_toy_pcm')
    
    pci_y_position_to_check =  0.9297
    
    reference_pci_x_position = 0.2041

    def T_minus_T_f(x):
    
        wval = w.leaf_node()(fenics.Point(x, pci_y_position_to_check))
        
        return wval[3] - T_f
    
    pci_x_position = opt.newton(T_minus_T_f, 0.01)
    
    assert(abs(pci_x_position - reference_pci_x_position) < 1.e-3)
    
    
if __name__=='__main__':

    test_melt_toy_pcm()
    