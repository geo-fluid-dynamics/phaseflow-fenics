import fenics


def apply_backward_euler(Delta_t, u):
        """ Apply the backward Euler (fully implicit, first order) time discretization method. """
        u_t = (u[0] - u[1])/Delta_t
        
        return u_t
        
    
def apply_bdf2(Delta_t, u):
    """ Apply the Gear/BDF2 (fully implicit, second order) backward difference formula for time discretization. 
    
    Use the constant time step size scheme from
    
        @article{belhamadia2012enhanced,
          title={An enhanced mathematical model for phase change problems with natural convection},
          author={Belhamadia, YOUSSEF and Kane, ABDOULAYE S and Fortin, ANDR{\'E}},
          journal={Int. J. Numer. Anal. Model},
          volume={3},
          number={2},
          pages={192--206},
          year={2012}
        }    
    """
    u_t = (3.*u[0] - 4.*u[1] + u[2])/(2.*Delta_t)
    
    return u_t
    