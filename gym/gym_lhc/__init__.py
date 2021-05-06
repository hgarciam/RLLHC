from gym.envs.registration import register

register(
    id='lhc-v0',
    entry_point='gym_lhc.envs:LHCEnv',
)
register(
    id='lhc-extrahard-v0',
    entry_point='gym_lhc.envs:LHCExtraHardEnv',
)