import fenics
import globals
import default


def initialize(W, parameters = default.parameters, m_B = default.m_B, ddtheta_m_B = default.ddtheta_m_B):
    # Define expressions needed for variational form
    Ra, Pr, K, g, gamma, mu_l = fenics.Constant(parameters['Ra']), fenics.Constant(parameters['Pr']), fenics.Constant(parameters['K']), fenics.Constant(parameters['g']), fenics.Constant(parameters['gamma']), fenics.Constant(parameters['mu_l'])
    
    Re = fenics.Constant(globals.Re)

    v, q, phi = fenics.TestFunctions(W)

    # Define variational form
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    def a(_mu, _u, _v):

        def D(_u):
        
            return sym(grad(_u))
        
        return 2.*_mu*inner(D(_u), D(_v))
        

    def b(_u, _q):
        
        return -div(_u)*_q
        

    def c(_w, _z, _v):
       
        return dot(dot(grad(_z), _w), _v)
        
        
    def make_nonlinear_form(dt = 1.e-3, w_n = fenics.Function(W)):
    
        w = fenics.Function(W)
        
        u, p, theta = fenics.split(w)
        
        u_n, p_n, theta_n = fenics.split(w_n)
    
        dt = fenics.Constant(dt)
        
        F = (
            b(u, q) - gamma*p*q
            + dot(u, v)/dt + c(u, u, v) + a(mu_l, u, v) + b(v, p) - dot(u_n, v)/dt + dot(m_B(theta)*g, v)
            + theta*phi/dt - dot(u, grad(phi))*theta + dot(K/Pr*grad(theta), grad(phi)) - theta_n*phi/dt
            )*fenics.dx
        
        return F

        
    w_w = fenics.TrialFunction(W)
    
    u_w, p_w, theta_w = fenics.split(w_w)

    def make_newton_linearized_form(dt = 1.e-3, w_k = fenics.Function(W), w_n = fenics.Function(W)):
    
        dt = fenics.Constant(dt)
        
        u_n, p_n, theta_n = fenics.split(w_n)
        
        u_k, p_k, theta_k = fenics.split(w_k)
        
        A = (
            b(u_w, q) - gamma*p_w*q
            + dot(u_w, v)/dt + c(u_w, u_k, v) + c(u_k, u_w, v) + a(mu_l, u_w, v) + b(v, p_w) + dot(ddtheta_m_B(theta_k)*theta_w*g, v)
            + theta_w*phi/dt - dot(u_k, grad(phi))*theta_w - dot(u_w, grad(phi))*theta_k + dot(K/Pr*grad(theta_w), grad(phi))
            )*fenics.dx
            
        L = (
            b(u_k, q) - gamma*p_k*q
            + dot(u_k - u_n, v)/dt + c(u_k, u_k, v) + a(mu_l, u_k, v) + b(v, p_k) + dot(m_B(theta_k)*g, v)
            + (theta_k - theta_n)*phi/dt - dot(u_k, grad(phi))*theta_k + dot(K/Pr*grad(theta_k), grad(phi))
            )*fenics.dx  
            
        return A, L
        
        
    return make_nonlinear_form, make_newton_linearized_form
    
    
if __name__=='__main__':

    initialize()