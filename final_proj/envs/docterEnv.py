import os
import importlib
import numpy as np
import gym
from gym import spaces
from final_proj.ple import PLE
from final_proj import docter

class DocterEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, normalize=False, display=False, **kwargs):
    self.game_name = 'BPWaveform'
    self.init(normalize, display, **kwargs)
    
  def init(self, normalize, display, **kwargs):
    game_module_name = 'final_proj.docter'
    game_module = importlib.import_module(game_module_name)
    self.game = getattr(game_module, self.game_name)(**kwargs)

    if display == False:
      # Do not open a PyGame window
      os.putenv('SDL_VIDEODRIVER', 'fbcon')
      os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    if normalize:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob_normalize, display_screen=display)
    else:
        self.gameOb = PLE(self.game, fps=30, state_preprocessor=self.get_ob, display_screen=display)
    
    self.viewer = None
    self.action_set = self.gameOb.getActionSet()
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.game.getGameState()),), dtype=np.float32)
    self.gameOb.init()

  def get_ob(self, state):
    return np.array(list(state.values()))

  def get_ob_normalize(self, state):
    state_normal = self.get_ob(state)
    # TODO
    return state_normal

  def step(self, action):
    reward = self.gameOb.act(self.action_set[action])
    done = self.gameOb.game_over()
    return (self.gameOb.getGameState(), reward, done, {})
    
  def reset(self):
    self.gameOb.reset_game()
    return self.gameOb.getGameState()
  
  def seed(self, seed=None):
    self.gameOb.rng.seed(seed)
    self.gameOb.init()
    return seed

  def render(self, mode='human'):
    
    img = self.gameOb.getScreenRGB() 
    #img = self.gameOb.getScreenGrayscale()
    #img = np.fliplr(np.rot90(self.gameOb.getScreenRGB(),3))
    print("here", img)
    if mode == 'rgb_array':

      return img


  def close(self):
    if self.viewer != None:
      self.viewer.close()
      self.viewer = None
    return 0

if __name__ == '__main__':
  env = DocterEnv(normalize=True)
  env.seed(0)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(10):
    ob = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      # env.render('rgb_array')
      env.render('human')
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()