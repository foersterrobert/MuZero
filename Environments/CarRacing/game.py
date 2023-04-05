import gymnasium as gym
import numpy as np
from PIL import Image

class CarRacing:
    def __init__(self, sequence_lenth=3, skip_frames=3, clip_reward=False, time_out_tolerance=100, time_out=25, render=False, eval=False):
        self.env = gym.make('CarRacing-v2', continuous=False, domain_randomize=False, render_mode='human' if render else 'rgb_array')
        self.action_size = self.env.action_space.n
        self.obs_stack = []
        self.sequence_lenth = sequence_lenth
        self.skip_frames = skip_frames
        self.clip_reward = clip_reward
        self.time_out_tolerance = time_out_tolerance
        self.time_out = time_out
        self.counter = 0
        self.negative_reward_counter = 0
        self.eval = eval

    def __repr__(self):
        return "CarRacing"

    def get_initial_state(self):
        self.counter = 0
        self.negative_reward_counter = 0
        observation, _ = self.env.reset()
        action, reward, is_terminal = 0, 0, False 
        observation = self.get_encoded_observation(observation, action)
        self.obs_stack = [observation for _ in range(self.sequence_lenth)]
        return np.stack(self.obs_stack), reward, is_terminal

    def step(self, action):
        reward = 0
        for _ in range(self.skip_frames + 1):
            observation, r, is_terminal, _, _ = self.env.step(action)
            reward += r
            if is_terminal:
                break
        observation = self.get_encoded_observation(observation, action)
        self.obs_stack = self.obs_stack[1:] + [observation]
        if self.clip_reward:
            reward = np.clip(reward, -float('inf'), 1)
        if reward < 0 and self.counter > self.time_out_tolerance and not self.eval:
            self.negative_reward_counter += 1
            if self.negative_reward_counter >= self.time_out:
                is_terminal = True
                self.env.close()
        else:
            self.counter += 1
        return np.stack(self.obs_stack), reward, is_terminal

    def get_encoded_observation(self, observation, action):
        obs = Image.fromarray(observation.copy())
        obs = obs.crop((0, 0, 96, 84))
        obs = obs.convert('L')
        obs = np.array(obs)
        actionPlane = np.full((12, 96), action * (255 / self.action_size), dtype=np.float32)
        obs = np.concatenate((obs, actionPlane), axis=0)
        obs /= 255
        return obs