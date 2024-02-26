from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np


class Env:
    """This class wraps PLE and Flappy Bird so that it behaves more like OpenAI Gym."""

    def __init__(self):
        # initializing the instance of FlappyBird class
        self.game = FlappyBird(pipe_gap=100)
        # then pass this object into PLE constructor and create an instance of that
        self.env = PLE(self.game, fps=30, display_screen=False)
        # init does some necessary things under the hood
        self.env.init()
        self.env.getGameState = self.game.getGameState  # maybe not necessary
        self.action_map = self.env.getActionSet()

    # function which takes an action
    def step(self, action):
        action = self.action_map[action]
        reward = self.env.act(action)
        done = self.env.game_over()
        obs = self.get_observation()
        return obs, reward, done

    def reset(self):
        self.env.reset_game()
        return self.get_observation()

    def get_observation(self):
        # game state returns a dictionary which describes
        # the meaning of each value
        # we only want the values
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def set_display(self, boolean_value):
        self.env.display_screen = boolean_value
