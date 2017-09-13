"""This module contains default values for some arguments/parameters.

Do not repeat yourself: With this module we can set such values 
in a single place.
"""
import fenics
import globals


parameters = {'Ra': 1.e6, 'Pr': 0.71, 'Ste': 0.045, 'C': 1, 'K': 1., 'g': (0., -1.), 'mu_l': 1., 'mu_s': 1.e8}

m_B = lambda theta : theta*parameters['Ra']/(parameters['Pr']*globals.Re**2)

ddtheta_m_B = lambda theta : parameters['Ra']/(parameters['Pr']*globals.Re**2)

'''Here we set an arbitrarily low theta_s to disable phase-change.'''
regularization = {'a_s': 2., 'theta_s': -1.e12, 'R_s': 0.005}

mesh = fenics.UnitSquareMesh(fenics.dolfin.mpi_comm_world(), 20, 20, 'crossed')

pressure_degree = 1

temperature_degree = 1

if __name__=='__main__':

    pass
    