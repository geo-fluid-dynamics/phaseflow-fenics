import phaseflow
import fenics
import tempfile


def test__cavity_freezing_simulation__ci__():

    sim = phaseflow.cavity_freezing_simulation.CavityFreezingSimulation(uniform_gridsize = 16, time_order = 2)
    
    sim.cold_wall_temperature_before_freezing.assign(0.25)
    
    sim.cold_wall_temperature_during_freezing.assign(-1.25)
      
    sim.timestep_size = 1.
    
    sim.temperature_rayleigh_number.assign(3.e5)
    
    sim.concentration_rayleigh_number.assign(-3.e4)
    
    sim.schmidt_number.assign(1.)
    
    sim.liquidus_slope.assign(-0.1)
    
    sim.regularization_smoothing_parameter.assign(1./64)
    
    sim.output_dir = tempfile.mkdtemp() + "test__cavity_freezing_simulation/"
    
    sim.run(endtime = 3., checkpoint_times = (None,))
    