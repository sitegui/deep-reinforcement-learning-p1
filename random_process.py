#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std


class OUProcess(RandomProcess):

    def __init__(self, size, theta, std):
        self.size = size
        self.theta = theta
        self.std = std
        self.reset_states()

    def reset_states(self):
        self.state = np.zeros(self.size)

    def sample(self):
        self.state += -self.theta * self.state + self.std * np.random.uniform(-1.0, 1.0, self.size)
        return self.state
