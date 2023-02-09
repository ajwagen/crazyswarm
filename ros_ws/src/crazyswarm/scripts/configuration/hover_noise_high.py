import numpy as np

from quadsim.learning.configuration.configuration import *

drone_config = DroneConfiguration(
    mass = ConfigValue[float](1.0, False),
    I = ConfigValue[float](1.0, False),
    g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind = False,
    pos = ConfigValue[np.ndarray](default=np.zeros(3), randomize=False),
    dir = ConfigValue[np.ndarray](default=np.zeros(3), randomize=False),
    vmax = ConfigValue[float](0, randomize=False)
)

init_config = InitializationConfiguration()

sim_config = SimConfiguration(
    linear_var=ConfigValue[float](default=0.1, randomize=False),
    angular_var=ConfigValue[float](default=0.1, randomize=False) 
)

config = AllConfig(drone_config, wind_config, init_config, sim_config)
