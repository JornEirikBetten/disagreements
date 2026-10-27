from envs.humanoid import Humanoid, make_humanoid_env
from envs.hopper import Hopper, make_hopper_env
from envs.walker2d import Walker2d, make_walker2d_env
from envs.ant import Ant, make_ant_env

LOCOMOTION_ENVS = ('humanoid', 'hopper', 'walker2d', 'ant')

__all__ = [
    'Humanoid',
    'make_humanoid_env',
    'Hopper',
    'make_hopper_env',
    'Walker2d',
    'make_walker2d_env',
    'Ant',
    'make_ant_env',
    'LOCOMOTION_ENVS',
]
