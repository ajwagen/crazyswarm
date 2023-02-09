import numpy as np

from quadsim.learning.configuration.configuration import *
from quadsim.learning.configuration.default_hover import drone_config, wind_config, sim_config

def init_random_box(possize, velsize):
    @dataclass
    class IC:
        pos: ConfigValue[np.ndarray] = ConfigValue[np.ndarray](
            default=np.array([0.0, 0.0, 0.0]),
            randomize=True,
            min=-possize * np.ones(3),
            max= possize * np.ones(3)
        )
        vel: ConfigValue[np.ndarray] = ConfigValue[np.ndarray](
            default=np.array([0.0, 0.0, 0.0]),
            randomize=True,
            min=-velsize * np.ones(3),
            max= velsize * np.ones(3)
        )
        # Represented as Euler ZYX in degrees
        rot: ConfigValue[np.ndarray] = ConfigValue[np.ndarray](
            default=np.array([0.0, 0.0, 0.0]),
            randomize=False
        )
        ang: ConfigValue[np.ndarray] = ConfigValue[np.ndarray](
            default=np.array([0.0, 0.0, 0.0]),
            randomize=False
        )

        sampler: Sampler = Sampler()

    return IC()

init_config = init_random_box(possize=0.1, velsize=0.1)

config = AllConfig(drone_config, wind_config, init_config, sim_config)
