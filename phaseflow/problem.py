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
    
    
    def make_nonlinear_form(self, dt, w_k, w_n, automatic_jacobian=True):
    
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
        
        mu_sl = lambda theta : heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)
        
        
        #
        w_w = fenics.TrialFunction(self.W)
        
        u_w, p_w, theta_w = fenics.split(w_w)
        
        v, q, phi = fenics.TestFunctions(self.W)
        
        u_k, p_k, theta_k = fenics.split(w_k)
        
        f_B = lambda theta : m_B(theta)*g

        F = (
            b(u_k, q) - gamma*p_k*q
            + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu_sl(theta_k), u_k, v) + b(v, p_k)
            + dot(f_B(theta_k), v)
            + C/dt*(theta_k - theta_n)*phi - dot(u_k, grad(phi))*C*theta_k + K/Pr*dot(grad(theta_k), grad(phi))
            + C/dt*(S(theta_k) - S(theta_n))*phi
            )*fenics.dx

        if automatic_jacobian:

            JF = fenics.derivative(F, w_k, w_w)
            
        else:
        
            ddtheta_m_B = self.ddtheta_m_B
            
            ddtheta_f_B = lambda theta : ddtheta_m_B(theta)*g
            
            ddtheta_heaviside_tanh = lambda theta, f_s, f_l: -(a_s*(fenics.tanh((a_s*(theta_s - theta))/R_s)**2 - 1.)*(f_l/2. - f_s/2.))/R_s
            
            ddtheta_S = lambda theta : ddtheta_heaviside_tanh(theta, f_s=S_s, f_l=S_l)
        
            ddtheta_mu_sl = lambda theta : ddtheta_heaviside_tanh(theta, f_s=mu_s, f_l=mu_l)        
        
            JF = (
                b(u_w, q) - gamma*p_w*q 
                + dot(u_w, v)/dt + c(u_k, u_w, v) + c(u_w, u_k, v) + a(theta_w*ddtheta_mu_sl(theta_k), u_k, v) + a(mu_sl(theta_k), u_w, v) + b(v, p_w) 
                + dot(theta_w*ddtheta_f_B(theta_k), v)
                + C/dt*theta_w*phi - dot(u_w, grad(phi))*C*theta_k - dot(u_k, grad(phi))*C*theta_w + K/Pr*dot(grad(theta_w), grad(phi))
                + C/dt*theta_w*ddtheta_S(theta_k)*phi
                )*fenics.dx

        
        return F, JF
        

        
class Problem(fenics.NonlinearProblem):

    def __init__(self, a, L, bcs):
    
        fenics.NonlinearProblem.__init__(self)
        
        self.a = a
        
        self.L = L
        
        self.bcs = bcs
        
    
    def F(self, b, x):
    
        assembler = fenics.SystemAssembler(self.a, self.L, self.bcs)
        
        assembler.assemble(b, x)
        
        
    def J(self, A, x):
    
        assembler = fenics.SystemAssembler(self.a, self.L, self.bcs)
        
        assembler.assemble(A)
        