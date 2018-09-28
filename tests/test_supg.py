import phaseflow
import fenics


def test_high_Ra_C_sea_water_freezing_with_supg():

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
    T_h_degC = 10.

    T_c_before_freezing_degC = 0.

    T_c_during_freezing_degC = -20.

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

    # Choose large time step size, grid spacing, and phase-interface smoothing large enough to quickly obtain preliminary results on a desktop computer.
    Delta_t = 1./32.

    h = 1./32.

    r = 1./64.


    # Set up the simulation.
    sim = phaseflow.cavity_freezing_simulation.CavityFreezingSimulation(
        time_order = 1,
        uniform_gridsize = round(1./h),
        stabilize_with_supg = True)

    sim.timestep_size = Delta_t

    sim.regularization_smoothing_parameter.assign(r)

    sim.prandtl_number.assign(Pr)

    sim.stefan_number.assign(Ste)

    sim.temperature_rayleigh_number.assign(Ra_T)

    sim.concentration_rayleigh_number.assign(Ra_C)

    sim.schmidt_number.assign(Sc)

    sim.initial_concentration.assign(C(C_0_percent_weight_NaCl))

    sim.liquidus_slope.assign(m_L)

    sim.hot_wall_temperature.assign(T(T_h_degC))

    sim.cold_wall_temperature_before_freezing.assign(T(T_c_before_freezing_degC))
            
    sim.cold_wall_temperature_during_freezing.assign(T(T_c_during_freezing_degC))


    # Run until $t = 1$.
    sim.run(endtime = 1., max_regularization_attempts = 16)
