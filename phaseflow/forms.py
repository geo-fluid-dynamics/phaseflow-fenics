import fenics
import globals
import default


inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym

def a(mu, u, v):

    def D(u):
    
        return sym(grad(u))
    
    return 2.*mu*inner(D(u), D(v))


def b(u, q):
    
    return -div(u)*q
    

def c(w, z, v):
   
    return dot(dot(grad(z), w), v)


class FormFactory():
    
    def __init__(self, W, parameters = default.parameters, m_B = default.m_B, ddtheta_m_B = default.ddtheta_m_B, regularization = default.regularization):
        
        self.parameters = parameters
        
        self.m_B = m_B
        
        self.ddtheta_m_B = ddtheta_m_B
        
        self.regularization = regularization
        
        self.W = W
        
        self.Re = fenics.Constant(globals.Re)
    
    
    def make_nonlinear_form(self, dt, w_, w_n, automatic_jacobian=True):
    
        dt = fenics.Constant(dt)
        
        Ra = fenics.Constant(self.parameters['Ra']), 
        
        Pr, Ste = fenics.Constant(self.parameters['Pr']), fenics.Constant(self.parameters['Ste'])
        
        C, K =  fenics.Constant(self.parameters['C']), fenics.Constant(self.parameters['K'])
        
        g, gamma = fenics.Constant(self.parameters['g']), fenics.Constant(self.parameters['gamma'])
        
        mu_l, mu_s = fenics.Constant(self.parameters['mu_l']), fenics.Constant(self.parameters['mu_s'])
        
        m_B = self.m_B
        
        a_s, theta_s = fenics.Constant(self.regularization['a_s']), fenics.Constant(self.regularization['theta_s'])
        
        R_s = fenics.Constant(self.regularization['R_s'])
        
        heaviside_tanh = lambda theta, f_s, f_l: f_l + (f_s - f_l)/2.*(1. + fenics.tanh(a_s*(theta_s - theta)/R_s))
        
        S_s = fenics.Constant(0.)
        
        S_l = fenics.Constant(1./Ste)
        
        S = lambda theta : heaviside_tanh(theta, f_s=S_s, f_l=S_l)
        
        u_n, p_n, theta_n = fenics.split(w_n)
        
        
        # @todo: Variable viscosity 
        
        #mu_sl = lambda theta : heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)
        
        mu_sl = fenics.Constant(mu_l)
        
        
        #
        dw = fenics.TrialFunction(self.W)
        
        du, dp, dtheta = fenics.split(dw)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        u_, p_, theta_ = fenics.split(w_)
        
        F = (
            b(u_, q) - gamma*p_*q
            + dot(u_, v)/dt + c(u_, u_, v) + a(mu_sl, u_, v) + b(v, p_)
            + dot(m_B(theta_)*g, v) 
            + C*(theta_ + S(theta_))*phi/dt - dot(u_, grad(phi))*C*theta_ + dot(K/Pr*grad(theta_), grad(phi)) 
            - dot(u_n, v)/dt 
            - C*(theta_n + S(theta_n))*phi/dt
            )*fenics.dx

        if automatic_jacobian:

            JF = fenics.derivative(F, w_, dw)
            
        else:
        
            ddtheta_m_B = self.ddtheta_m_B
            
            ddtheta_heaviside_tanh = lambda theta, f_s, f_l: -(a_s*(fenics.tanh((a_s*(theta_s - theta))/R_s)**2 - 1.)*(f_l/2. - f_s/2.))/R_s
            
            ddtheta_S = lambda theta : ddtheta_heaviside_tanh(theta, f_s=S_s, f_l=S_l)   
        
            # @todo: Variable viscosity 
            
            #ddtheta_mu_sl = lambda theta : ddtheta_heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)        
        
            ddtheta_mu_sl = fenics.Constant(0.)
        
            JF = (
                b(du, q) - gamma*dp*q 
                + dot(du, v)/dt + c(u_, du, v) + c(du, u_, v) + a(dtheta*ddtheta_mu_sl, u_, v) + a(mu_sl, du, v) + b(v, dp) 
                + dot(dtheta*ddtheta_m_B(theta_)*g, v)
                + C*(dtheta*(1 + ddtheta_S(theta_)))*phi/dt + dot(du, grad(phi))*C*theta_ + dot(u_, grad(phi))*dtheta + K/Pr*dot(grad(dtheta), grad(phi))
                )*fenics.dx

        
        return F, JF
        