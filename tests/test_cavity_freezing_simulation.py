import phaseflow
import fenics
import tempfile


def test__cavity_freezing_simulation__ci__():

    sim = phaseflow.cavity_freezing_simulation.CavityFreezingSimulation(uniform_gridsize = 16, time_order = 2)
    
    
    sim.cold_wall_temperature_before_freezing.assign(0.25)
    
    sim.cold_wall_temperature_during_freezing.assign(-1.25)
    
    sim.temperature_rayleigh_number.assign(3.e5)
    
    sim.concentration_rayleigh_number.assign(-3.e4)
    
    sim.schmidt_number.assign(1.)
    
    sim.liquidus_slope.assign(-0.1)
    
    sim.regularization_smoothing_parameter.assign(1./16.)
    
    
    sim.output_dir = tempfile.mkdtemp() + "/test__cavity_freezing_simulation/"
    
    
    sim.timestep_size.assign(1.)
    
    endtime = 3.
    
    sim.run(endtime = endtime, checkpoint_times = (0., endtime,))
    
    A_S = fenics.assemble(sim.solid_area_integrand())
    
    assert(abs(A_S - 0.177) < 1.e-3)
    