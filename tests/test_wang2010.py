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


def verify_against_wang2010(w, mesh):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
        
            ux = wval[0]*data['Pr']/data['Ra']**0.5
        
            assert(abs(ux - true_ux) < 2.e-2)
            
    
output_dir='output/test_wang2010_natural_convection_air/'

@pytest.mark.dependency()
def test_wang2010_natural_convection_air():
    
    w, mesh = phaseflow.run(output_dir=output_dir, stop_when_steady=True)
        
    verify_against_wang2010(w, mesh)
    

@pytest.mark.dependency(depends=["test_wang2010_natural_convection_air"])
def test_wang2010_natural_convection_air_restart():

    w, mesh = phaseflow.run(restart = True,
        restart_filepath = output_dir+'restart_t0.067.h5',
        start_time = 0.067,
        output_dir=output_dir)
        
    verify_against_wang2010(w, mesh)
    
    
if __name__=='__main__':
    
    test_wang2010_natural_convection_air()
    
    test_wang2010_natural_convection_air_restart()
