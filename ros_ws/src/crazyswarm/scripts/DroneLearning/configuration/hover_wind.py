import numpy as np

from quadsim.learning.configuration.configuration import *

drone_config = DroneConfiguration(
    mass = ConfigValue[float](1.0, False),
    I = ConfigValue[float](1.0, False),
    g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind = True,
    pos = ConfigValue[np.ndarray](default=np.zeros(3), randomize=False),
    dir = ConfigValue[np.ndarray](
        default=np.array([3, 3, 3]), 
        randomize=False,
        min=np.array([-4, -4, -4]),
        max=np.array([4, 4, 4])
    ),
    vmax = ConfigValue[float](500, randomize=False)
)

init_config = InitializationConfiguration()

sim_config = SimConfiguration(
    linear_var=ConfigValue[float](default=0.0, randomize=False),
    angular_var=ConfigValue[float](default=0.0, randomize=False),
    obs_noise=ConfigValue[float](default=0.005, randomize=False)
)

adapt_config = AdaptationConfiguration(
    include = []
)

config = AllConfig(drone_config, wind_config, init_config, sim_config, adapt_config)
