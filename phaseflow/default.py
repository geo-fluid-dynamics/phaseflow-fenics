import fenics
import globals


parameters = {'Ra': 1.e6, 'Pr': 0.71, 'Ste': 0.045, 'C': 1, 'K': 1., 'g': (0., -1.), 'mu_l': 1.}

m_B = lambda theta : theta*parameters['Ra']/(parameters['Pr']*globals.Re**2)

ddtheta_m_B = lambda theta : parameters['Ra']/(parameters['Pr']*globals.Re**2)

regularization = {'a_s': 2., 'theta_s': 0.01, 'epsilon_1': 0.01, 'epsilon_2': 0.01, 'R_s': 0.005}

mesh = fenics.UnitSquareMesh(20, 20, 'crossed')

pressure_degree = 1

temperature_degree = 1


if __name__=='__main__':

    pass