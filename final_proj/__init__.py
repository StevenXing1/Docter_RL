from gym.envs.registration import register
register(
  id='final_proj/RLDocter_v0',
  entry_point='final_proj.envs:DocterEnv'
)