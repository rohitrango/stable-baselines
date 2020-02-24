import warnings

import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines.common.running_mean_std import RunningMeanStd


class VecAtari(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.

    It is pickleable which will save moving averages and configuration parameters.
    The wrapped environment `venv` is not saved, and must be restored manually with
    `set_venv` after being unpickled.

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv,):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        if len(obs.shape) == 4:
            obs = obs[0].transpose(2, 0, 1)
        else:
            obs = obs.tranpose(2, 0, 1)
        return obs, rews, news, infos

    def get_original_obs(self):
        """
        returns the unnormalized observation

        :return: (numpy float)
        """
        if len(self.old_obs.shape) == 4:
            old_obs = self.old_obs[0].transpose(2, 0, 1)
        else:
            old_obs = self.old_obs.transpose(2, 0, 1)
        return old_obs


    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        if len(obs.shape) == 4:
            obs = obs[0].transpose(2, 0, 1)
        else:
            obs = obs.tranpose(2, 0, 1)
        return obs

