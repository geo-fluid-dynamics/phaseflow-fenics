"""This module contains the variational form.

This is where the entire mathematical model is described.
"""
import fenics
import globals
import default



class FormFactory():
    """This class provides a method to create the nonlinear form.
    
    This is needed for example when the time step size changes
    or when the grid is refined.
    """
    
    """@todo This approach was used before discovering that FEniCS allows
    one to use dt.assign() to change dt and automatically use the
    updated value during matrix assembly."""
    def __init__(self, W, parameters = default.parameters, m_B = default.m_B, ddT_m_B = default.ddT_m_B, regularization = default.regularization):
        
        self.parameters = parameters
        
        self.m_B = m_B
        
        self.ddT_m_B = ddT_m_B
        
        self.regularization = regularization
        
        self.W = W
        
        self.Re = fenics.Constant(globals.Re)
    
    
    def make_nonlinear_form(self, dt, w_k, w_n, automatic_jacobian=True):
        """Create the nonlinear form.
        
        The result is a fenics form to be handed 
        to NonlinearVariationalProblem.
        """
    
        
        
        return F, JF, M

        
if __name__=='__main__':

    pass
    