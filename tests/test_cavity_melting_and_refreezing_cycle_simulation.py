import phaseflow
import fenics


# ## Environment
g = 9.80665  # [m/s^2]


# ## Material properties

# Sea water at zero degrees C per [NPL](http://www.kayelaby.npl.co.uk/general_physics/2_7/2_7_9.html)
nu = 1.83e-6  # [m^2/s]

alpha = 1.37e-7  # [m^2/s]

beta_T_degC = 8.1738e-5  # [(deg C)^-1]

beta_C_percent_weight_NaCl = -7.8284e-3  # [(% wt. NaCl)^-1]


# Sea water as a eutectic binary alloy per Worster
Le = 80.

T_E_degC = -21.1

C_E_percent_weight_NaCl = 23.3


# Pure water ice, per \cite{Lide2010} (from Kai's thesis)
c_p = 2110.0  # [J / (kg K)]

h_m = 333641.9  # [J / kg]


# ## Domain, initial values, and boundary conditions

# Square cavity from \cite{mikalek2003simulations}
L = 0.038  # [m]


# Salinity [typical of Earth's oceans](https://en.wikipedia.org/wiki/Seawater)
C_0_percent_weight_NaCl = 3.5


# Choose boundary conditions that will cause strong convection and quickly induce freezing.
T_h_during_melting_degC = 20.

T_h_during_freezing_degC = 0.

T_c_during_melting_degC = -10.

T_c_during_freezing_degC = -20.

T_h_degC = T_h_during_melting_degC

T_c_degC = T_c_during_freezing_degC


# ## Scaling
T_ref_degC = 0.

def T(T_degC):
    
    return (T_degC - T_ref_degC)/(T_h_degC - T_c_degC)

def C(C_percent_weight_NaCl):
    
    return C_percent_weight_NaCl/C_0_percent_weight_NaCl


# ## Derived parameters
Ste = c_p*(T_h_degC - T_c_degC)/h_m

Pr = nu/alpha

D = alpha/Le

Sc = nu/D

Gr_T = g*L**3/nu**2*beta_T_degC*(T_h_degC - T_c_degC)

Ra_T = Gr_T*Pr

Gr_C = g*L**3/nu**2*beta_C_percent_weight_NaCl*C_0_percent_weight_NaCl

Ra_C = Gr_C*Pr


# Calculate the dimensionless liquidus slope based on the eutectic point.
m_L = (T(T_E_degC) - 0.)/(C(C_E_percent_weight_NaCl) - 0.)


# # Simulation
def run(
        T_h_during_melting = T(T_h_during_melting_degC),
        T_h_during_freezing = T(T_h_during_freezing_degC),
        T_c_during_melting = T(T_c_during_melting_degC),
        T_c_during_freezing = T(T_c_during_freezing_degC),
        Pr = Pr,
        Ste = Ste,
        Ra_T = Ra_T,
        Ra_C = -1.e6,
        Sc = Sc,
        C_0 = C(C_0_percent_weight_NaCl),
        m_L = m_L,
        Delta_t = 1./8.,
        M = 20,
        r = 1./128.,
        melting_endtime = 1.,
        freezing_endtime = 1.,
        checkpoint_times = (1./4., 1./2., 1., 2., 3., 4., 5., 6., 8., 10.),
        time_order = 1,
        quadrature_degree = 3,
        max_regularization_attempts = 16,
        max_newton_iterations = 50):

    sim = phaseflow.cavity_melting_and_refreezing_cycle_simulation.        CavityMeltingAndRefreezingCycleSimulation(
            time_order = time_order,
            uniform_gridsize = M,
            quadrature_degree = quadrature_degree)
    
    sim.solver.parameters["newton_solver"]["maximum_iterations"] =         max_newton_iterations
    
    sim.output_dir = "MeltingAndRefreezing/"
    
    sim.timestep_size.assign(Delta_t)

    sim.regularization_smoothing_parameter.assign(r)

    sim.prandtl_number.assign(Pr)

    sim.stefan_number.assign(Ste)

    sim.temperature_rayleigh_number.assign(Ra_T)

    sim.concentration_rayleigh_number.assign(Ra_C)

    sim.schmidt_number.assign(Sc)

    sim.initial_concentration.assign(C_0)

    sim.liquidus_slope.assign(m_L)

    sim.hot_wall_temperature_during_melting.assign(T_h_during_melting)
    
    sim.hot_wall_temperature_during_freezing.assign(T_h_during_freezing)

    sim.cold_wall_temperature_during_melting.assign(T_c_during_melting)

    sim.cold_wall_temperature_during_freezing.assign(T_c_during_freezing)
    
    sim.run(
        melting_endtime = melting_endtime, 
        freezing_endtime = freezing_endtime,
        checkpoint_times = checkpoint_times, 
        plot = True,
        savefigs = True,
        max_regularization_attempts = max_regularization_attempts)


run()

