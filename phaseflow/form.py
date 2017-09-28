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
        f_B = lambda T : m_B(T)*g
        
        """Parameter shifting the tanh regularization"""
        T_f = fenics.Constant(self.regularization['T_f'])
        
        """Parameter scaling the tanh regularization"""
        r = fenics.Constant(self.regularization['r'])
        
        """Latent heat"""
        L = C/Ste
        
        """Regularize heaviside function with a 
        hyperoblic tangent function."""
        P = lambda T: 0.5*(1. - fenics.tanh(2.*(T_f - T)/r))
        
        """Variable viscosity"""
        mu = lambda (T) : mu_s + (mu_l - mu_s)*P(T)
        
        """Set the nonlinear variational form."""
        u_n, p_n, T_n = fenics.split(w_n)

        w_w = fenics.TrialFunction(self.W)
        
        u_w, p_w, T_w = fenics.split(w_w)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        u_k, p_k, T_k = fenics.split(w_k)

        F = (
            b(u_k, q) - gamma*p_k*q
            + dot(u_k - u_n, v)/dt
            + c(u_k, u_k, v) + b(v, p_k) + a(mu(T_k), u_k, v)
            + dot(f_B(T_k), v)
            + C/dt*(T_k - T_n)*phi
            - dot(C*T_k*u_k, grad(phi)) 
            + K/Pr*dot(grad(T_k), grad(phi))
            + 1./dt*L*(P(T_k) - P(T_n))*phi
            )*fenics.dx

        if automatic_jacobian:

            JF = fenics.derivative(F, w_k, w_w)
            
        else:
        
            ddT_m_B = self.ddT_m_B
            
            ddT_f_B = lambda T : ddT_m_B(T)*g
            
            sech = lambda theta: 1./fenics.cosh(theta)
            
            dP = lambda T: sech(2.*(T_f - T)/r)**2/r
        
            dmu = lambda T : (mu_l - mu_s)*dP(T)

            """Set the Jacobian (formally the Gateaux derivative)."""
            JF = (
                b(u_w, q) - gamma*p_w*q 
                + dot(u_w, v)/dt
                + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
                + a(T_w*dmu(T_k), u_k, v) + a(mu(T_k), u_w, v) 
                + dot(T_w*ddT_f_B(T_k), v)
                + C/dt*T_w*phi
                - dot(C*T_k*u_w, grad(phi))
                - dot(C*T_w*u_k, grad(phi))
                + K/Pr*dot(grad(T_w), grad(phi))
                + 1./dt*L*T_w*dP(T_k)*phi
                )*fenics.dx

        return F, JF

        
if __name__=='__main__':

    pass
    