from .context import phaseflow

import fenics
import pytest


def verify_against_wang2010(w, mesh):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
        
            ux = wval[0]*data['Pr']/data['Ra']**0.5
        
            assert(abs(ux - true_ux) < 2.e-2)
        
        
def wang2010_natural_convection_air(output_dir='output/test_wang2010_natural_convection_air',
        start_time=0., end_time=1., restart=False, output_times=('start', 1.e-3, 0.01, 0.1, 'end',),
        automatic_jacobian=True):

    m = 20
    
    theta_hot = 0.5
    
    theta_cold = -0.5
    
    theta_s = theta_cold - 1.
    
    w, mesh = phaseflow.run(
        Ste = 1.e16,
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        time_step_bounds = (1.e-3, 1.e-3, 0.01),
        start_time = start_time,
        end_time = end_time,
        output_times = output_times,
        stop_when_steady = True,
        automatic_jacobian = automatic_jacobian,
        custom_newton = False,
        regularization = {'a_s': 2., 'theta_s': theta_s, 'R_s': 0.1*abs(theta_s)},
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta_hot)+"*near(x[0],  0.) + "+str(theta_cold)+"*near(x[0],  1.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_hot), 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_cold), 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        output_dir = output_dir,
        debug = True,
        restart = restart,
        restart_filepath = output_dir+'/restart_t'+str(start_time)+'.h5')

    return w, mesh
    
    
def test_debug_wang2010_natural_convection_air_autoJ():
    
    w, mesh = wang2010_natural_convection_air(automatic_jacobian=True)
        
    verify_against_wang2010(w, mesh)
    
    
@pytest.mark.dependency()
def test_wang2010_natural_convection_air_manualJ():
    
    w, mesh = wang2010_natural_convection_air(end_time = 10., automatic_jacobian=False)
        
    verify_against_wang2010(w, mesh)
    

@pytest.mark.dependency(depends=["test_wang2010_natural_convection_air_manualJ"])
def test_wang2010_restart():

    w, mesh = wang2010_natural_convection_air(start_time = 0.1, output_times = ('start', 'end'),
        restart=True, automatic_jacobian=False)
        
    verify_against_wang2010(w, mesh)
    
    
if __name__=='__main__':

    test_debug_wang2010_natural_convection_air_autoJ()
    
    test_wang2010_natural_convection_air_manualJ()
    
    test_wang2010_restart()
