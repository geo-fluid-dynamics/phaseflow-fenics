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
    
    
    def make_nonlinear_form(self, dt, w_k, w_n):
    
        dt = fenics.Constant(dt)
        
        Ra, Pr, Ste, C, K, g, gamma, mu_l = fenics.Constant(self.parameters['Ra']), fenics.Constant(self.parameters['Pr']), fenics.Constant(self.parameters['Ste']), fenics.Constant(self.parameters['C']), fenics.Constant(self.parameters['K']), fenics.Constant(self.parameters['g']), fenics.Constant(self.parameters['gamma']), fenics.Constant(self.parameters['mu_l'])
        
        m_B = self.m_B
        
        a_s, theta_s, R_s = fenics.Constant(self.regularization['a_s']), fenics.Constant(self.regularization['theta_s']), fenics.Constant(self.regularization['R_s'])
        
        heaviside_tanh = lambda theta, f_s, f_l: f_l + (f_s - f_l)/2.*(1. + fenics.tanh(a_s*(theta_s - theta)/R_s))
        
        S_s = fenics.Constant(0.)
        
        S_l = fenics.Constant(1./Ste)
        
        S = lambda theta : heaviside_tanh(theta, f_s=S_s, f_l=S_l)
        
        u_n, p_n, theta_n = fenics.split(w_n)
        
        
        #
        w = fenics.TrialFunction(self.W)
        
        u, p, theta = fenics.split(w)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        F = (
            b(u, q) - gamma*p*q
            + dot(u, v)/dt + c(u, u, v) + a(mu_l, u, v) + b(v, p) + dot(m_B(theta)*g, v) - dot(u_n, v)/dt
            + C*theta*phi/dt - dot(u, grad(phi))*C*theta + dot(K/Pr*grad(theta), grad(phi)) - C*theta_n*phi/dt 
            + C*S(theta)*phi/dt - S(theta_n)*phi/dt
            )*fenics.dx

        
        # Jacobian per http://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/nonlinear.html
        
        #w_w = fenics.TrialFunction(self.W)
        
        R = fenics.action(F, w_k)
        
        JR = fenics.derivative(R, w_k)
        
        '''
        ddtheta_m_B = self.ddtheta_m_B
        
        ddtheta_heaviside_tanh = lambda theta, f_s, f_l: -(a_s*(fenics.tanh((a_s*(theta_s - theta))/R_s)**2 - 1.)*(f_l/2. - f_s/2.))/R_s
        
        ddtheta_S = lambda theta : ddtheta_heaviside_tanh(theta, f_s=S_s, f_l=S_l)   
        
        ddtheta_mu_l = 0. # @todo Add variable viscosity
        
        u_k, p_k, theta_k = fenics.split(w_k)
        
        JR = (
            b(u, q) - gamma*p*q
            - (b(u_k, q) - gamma*p_k*q)
            + dot(u, v)/dt + c(u, u_k, v) + c(u_k, u, v) + a(mu_l, u, v) + a(ddtheta_mu_l*theta, u_k, v) + b(v, p) + dot(ddtheta_m_B(theta_k)*theta*g, v) 
            - (dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu_l, u_k, v) + b(v, p_k) + dot(m_B(theta_k)*g, v))
            + theta*phi/dt - dot(u_k, grad(phi))*theta - dot(u, grad(phi))*theta_k + dot(K/Pr*grad(theta), grad(phi))
            - ((theta_k - theta_n)*phi/dt - dot(u_k, grad(phi))*theta_k + dot(K/Pr*grad(theta_k), grad(phi)))
            + dot(ddtheta_S(theta_k)*theta, phi)/dt
            - ((S(theta_k) - S(theta_n))*phi/dt)
            )*fenics.dx
        '''
        
        return R, JR
        