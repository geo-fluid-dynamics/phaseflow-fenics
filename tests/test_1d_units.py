from .context import phaseflow

import fenics
        
        
def test_1d_output():

    T_h = 1.
    
    T_c = -1.
    
    w = phaseflow.run(
        output_dir = 'output/test_1D_output/',
        prandtl_number = 1.,
        stefan_number = 1.,
        gravity = [0.],
        mesh = fenics.UnitIntervalMesh(1000),
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(T_h)+" - "+str(T_c)+")*near(x[0],  0.) "+str(T_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = 0.01,
        regularization_smoothing_factor = 0.005,
        end_time = 0.001,
        time_step_size = 0.001)

        
def test_1d_velocity():

    mesh = fenics.UnitIntervalMesh(fenics.dolfin.mpi_comm_world(), 5)

    V = fenics.VectorFunctionSpace(mesh, 'P', 1)

    u = fenics.Function(V)

    bc = fenics.DirichletBC(V, [10.0], 'x[0] < 0.5')

    print(bc.get_boundary_values())
    

if __name__=='__main__':
    
    test_1d_output()
    
    test_1d_velocity()
