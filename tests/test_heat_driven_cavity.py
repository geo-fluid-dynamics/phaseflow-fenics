"""This module tests the natural convection of air.

This verifies against the results published in 

@article{wang2010comprehensive,
  title={A comprehensive numerical model for melting with natural convection},
  author={Wang, Shimin and Faghri, Amir and Bergman, Theodore L},
  journal={International Journal of Heat and Mass Transfer},
  volume={53},
  number={9},
  pages={1986--2000},
  year={2010},
  publisher={Elsevier}
}

phaseflow runs this case as a minimum working example by default,
so see the default arguments of phaseflow.core.run() for the 
problem setup.
"""
from .context import phaseflow

import fenics
import pytest


def verify_against_wang2010(w):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    bbt = w.function_space().mesh().bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
        
            ux = wval[1]*data['Pr']/data['Ra']**0.5
        
            assert(abs(ux - true_ux) < 2.e-2)
            
    
def heat_driven_cavity(output_dir = "output/heat_driven_cavity",
        time = 0.,
        initial_values = None):

    T_hot = 0.5
    
    T_cold = -T_hot
    
    def m_B(T, Ra, Pr, Re):
        
        return T*Ra/(Pr*Re**2)

        
    def ddT_m_B(T, Ra, Pr, Re):

        return Ra/(Pr*Re**2)
        
        
    if initial_values is None:
    
        mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), 20, 20, "crossed")
        
        mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
        W = fenics.FunctionSpace(mesh, mixed_element)
          
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "T_hot + x[0]*(T_cold - T_hot)"),
                T_hot = T_hot, T_cold = T_cold,
                element = mixed_element),
            W)
    
    else:
    
        W = initial_values.function_space()
        
    solution = fenics.Function(W)
    
    phaseflow.run(solution,
        initial_values = initial_values,
        time = time,
        boundary_conditions = [
            fenics.DirichletBC(W.sub(1), (0., 0.),
                "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"),
            fenics.DirichletBC(W.sub(2), T_hot, "near(x[0],  0.)"),
            fenics.DirichletBC(W.sub(2), T_cold, "near(x[0],  1.)")],
        output_dir=output_dir,
        time_step_size = 1.e-3,
        end_time = 10.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        gravity = (0., -1.),
        m_B = m_B,
        ddT_m_B = ddT_m_B,
        stop_when_steady = True,
        steady_relative_tolerance=1.e-4)
                
    return solution, time
    
    
@pytest.mark.dependency()
def test_heat_driven_cavity():
    
    solution, time = heat_driven_cavity(output_dir="output/test_heat_driven_cavity")
        
    verify_against_wang2010(solution)
    

@pytest.mark.dependency(depends=["test_heat_driven_cavity"])
def test_heat_driven_cavity_restart():

    solution, time = phaseflow.read_checkpoint("output/test_heat_driven_cavity/checkpoint_t0.06.h5")
    
    solution, time = heat_driven_cavity(initial_values = solution,
        time = time,
        output_dir="output/test_heat_driven_cavity/restart_t0.06/")
        
    verify_against_wang2010(solution)
    
    
if __name__=='__main__':
    
    test_heat_driven_cavity()
    
    test_heat_driven_cavity_restart()
