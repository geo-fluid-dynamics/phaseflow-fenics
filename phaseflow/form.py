"""This module contains the variational form.

This is where the entire mathematical model is described.
"""
import fenics
import globals
import default


"""Set local names for math operators to improve readability."""
inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym


"""Next we write the linear, bilinear, and trilinear forms.

These follow the common notation for applying the finite element method
to the incompressible Navier-Stokes equations, e.g. from danaila2014newton
and huerta2003fefluids.
"""

"""The bilinear form for the stress-strain matrix in Stokes flow."""
def a(mu, u, v):

    def D(u):
    
        return sym(grad(u))
    
    return 2.*mu*inner(D(u), D(v))


"""The linear form for the divergence in incompressible flow."""
def b(u, q):
    
    return -div(u)*q
    

"""The trilinear form for convection of the velocity field."""
def c(w, z, v):
   
    return dot(dot(grad(z), w), v)


class FormFactory():
    """This class provides a method to create the nonlinear form.
    
    This is needed for example when the time step size changes
    or when the grid is refined.
    """
    
    """@todo This approach was used before discovering that FEniCS allows
    one to use dt.assign() to change dt and automatically use the
    updated value during matrix assembly."""
    def __init__(self, W, parameters = default.parameters, m_B = default.m_B, ddtheta_m_B = default.ddtheta_m_B, regularization = default.regularization):
        
        self.parameters = parameters
        
        self.m_B = m_B
        
        self.ddtheta_m_B = ddtheta_m_B
        
        self.regularization = regularization
        
        self.W = W
        
        self.Re = fenics.Constant(globals.Re)
    
    
    def make_nonlinear_form(self, dt, w_k, w_n, automatic_jacobian=True):
        """Create the nonlinear form.
        
        The result is a fenics form to be handed 
        to NonlinearVariationalProblem.
        """
    
        """Time step size."""
        dt = fenics.Constant(dt)
        
        """Rayleigh Number"""
        Ra = fenics.Constant(self.parameters['Ra']), 
        
        """Prandtl Number"""
        Pr = fenics.Constant(self.parameters['Pr'])
        
        """Stefan Number"""
        Ste = fenics.Constant(self.parameters['Ste'])
        
        """Heat capacity"""
        C = fenics.Constant(self.parameters['C'])
        
        """Thermal diffusivity"""
        K = fenics.Constant(self.parameters['K'])
        
        """Gravity"""
        g = fenics.Constant(self.parameters['g'])
        
        """Parameter for penalty formulation
        of incompressible Navier-Stokes"""
        gamma = fenics.Constant(self.parameters['gamma'])
        
        """Liquid viscosity"""
        mu_l = fenics.Constant(self.parameters['mu_l'])
        
        """Solid viscosity"""
        mu_s = fenics.Constant(self.parameters['mu_s'])
        
        """Density function for buoyancy force term"""
        m_B = self.m_B
        
        """Buoyancy force, $f = ma$"""
        f_B = lambda theta : m_B(theta)*g
        
        """Parameter shifting the tanh regularization"""
        theta_s = fenics.Constant(self.regularization['theta_s'])
        
        """Parameter scaling the tanh regularization"""
        R_s = fenics.Constant(self.regularization['R_s'])
        
        # @todo Remove a_s; it's a redundant parameter.
        a_s = fenics.Constant(self.regularization['a_s'])
        
        """Regularize heaviside function with a 
        hyperoblic tangent function."""
        heaviside_tanh = lambda theta, f_s, f_l: f_l + (f_s - f_l)/2.*(1. + fenics.tanh(a_s*(theta_s - theta)/R_s))
        
        """Variable viscosity"""
        mu = lambda theta : heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)
        
        """Enthalpy source/sink from latent heat"""
        S_s = fenics.Constant(0.)
        
        S_l = fenics.Constant(1./Ste)
        
        S = lambda theta : heaviside_tanh(theta, f_s=S_s, f_l=S_l)

        
        """Set the nonlinear variational form."""
        u_n, p_n, theta_n = fenics.split(w_n)

        w_w = fenics.TrialFunction(self.W)
        
        u_w, p_w, theta_w = fenics.split(w_w)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        u_k, p_k, theta_k = fenics.split(w_k)

        F = (
            b(u_k, q) - gamma*p_k*q
            + dot(u_k - u_n, v)/dt
            + c(u_k, u_k, v) + b(v, p_k) + a(mu(theta_k), u_k, v)
            + dot(f_B(theta_k), v)
            + C/dt*(theta_k - theta_n)*phi
            - dot(C*theta_k*u_k, grad(phi)) + K/Pr*dot(grad(theta_k), grad(phi))
            + C/dt*(S(theta_k) - S(theta_n))*phi
            )*fenics.dx

        if automatic_jacobian:

            JF = fenics.derivative(F, w_k, w_w)
            
        else:
        
            ddtheta_m_B = self.ddtheta_m_B
            
            ddtheta_f_B = lambda theta : ddtheta_m_B(theta)*g
            
            ddtheta_heaviside_tanh = lambda theta, f_s, f_l: -(a_s*(fenics.tanh((a_s*(theta_s - theta))/R_s)**2 - 1.)*(f_l/2. - f_s/2.))/R_s
            
            dS = lambda theta : ddtheta_heaviside_tanh(theta, f_s=S_s, f_l=S_l)
        
            dmu = lambda theta : ddtheta_heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)        

            """Set the Jacobian (formally the Gateaux derivative)."""
            JF = (
                b(u_w, q) - gamma*p_w*q 
                + dot(u_w, v)/dt
                + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
                + a(theta_w*dmu(theta_k), u_k, v) + a(mu(theta_k), u_w, v) 
                + dot(theta_w*ddtheta_f_B(theta_k), v)
                + C/dt*theta_w*phi
                - dot(C*theta_k*u_w, grad(phi))
                - dot(C*theta_w*u_k, grad(phi))
                + K/Pr*dot(grad(theta_w), grad(phi))
                + C/dt*theta_w*dS(theta_k)*phi
                )*fenics.dx

        return F, JF

        
if __name__=='__main__':

    pass
    