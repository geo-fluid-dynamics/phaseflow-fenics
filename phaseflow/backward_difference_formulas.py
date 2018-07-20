import fenics


def apply_backward_euler(Delta_t, u):
    """ Apply the backward Euler (fully implicit, first order) time discretization method. """
    u_t = (u[0] - u[1])/Delta_t
    
    return u_t
    
    
def apply_bdf2(Delta_t, u):
    """ Apply the Gear/BDF2 (fully implicit, second order) backward difference formula for time discretization. 
    
    Here we use the variable time step size formula given by equation (12) from
    
        @article{eckert2004bdf2,
          title={A BDF2 integration method with step size control for elasto-plasticity},
          author={Eckert, S and Baaser, H and Gross, D and Scherf, O},
          journal={Computational Mechanics},
          volume={34},
          number={5},
          pages={377--386},
          year={2004},
          publisher={Springer}
        }
        
    which we interpreted in the context of finite difference time discretizations in a PDE,
    rather than as an ODE solver, by comparing to the constant time step size BDF2 formula in
    
        @article{belhamadia2012enhanced,
          title={An enhanced mathematical model for phase change problems with natural convection},
          author={Belhamadia, YOUSSEF and Kane, ABDOULAYE S and Fortin, ANDR{\'E}},
          journal={Int. J. Numer. Anal. Model},
          volume={3},
          number={2},
          pages={192--206},
          year={2012}
        }    
        
    Perhaps we could find a reference which derives variable time step size BDF2 
    in the context of finite difference time discretization,
    rather than in the context of ODE solvers.
    """
    tau = Delta_t[0]/Delta_t[1]
    
    u_t = 1./Delta_t[0]*((1. + 2.*tau)/(1. + tau)*u[0] - (1. + tau)*u[1] + tau*tau/(1. + tau)*u[2])
    
    return u_t
    