import numpy as np

from quadsim.learning.configuration.configuration import *

drone_config = DroneConfiguration(
    mass = ConfigValue[float](1.0, randomize=True, min=0.5, max=2.0),
    I = ConfigValue[float](1.0, randomize=False, min=0.5, max=2.0),
    g = ConfigValue[float](9.8, False)
)

wind_config = WindConfiguration(
    is_wind = True,
    pos = ConfigValue[np.ndarray](default=np.zeros(3), randomize=False),
    dir = ConfigValue[np.ndarray](
        default=np.zeros(3), 
        randomize=True,
        min=np.array([-4, -4, -4]),
        max=np.array([4, 4, 4])
    ),
    vmax = ConfigValue[float](500, randomize=False)
)

init_config = InitializationConfiguration()

sim_config = SimConfiguration(
    linear_var=ConfigValue[float](default=0.0, randomize=False),
    angular_var=ConfigValue[float](default=0.0, randomize=False),
    obs_noise=ConfigValue[float](default=0.005, randomize=False),
    latency=ConfigValue[int](default=0, randomize=False, min=0.01, max=0.05),
    k=ConfigValue[float](default=1, randomize=False)
)

adapt_config = AdaptationConfiguration(
    include = [EnvCondition.WIND, EnvCondition.MASS]
)

config = AllConfig(drone_config, wind_config, init_config, sim_config, adapt_config)
