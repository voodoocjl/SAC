import numpy as np
import random

# key parameters of environment
max_episode =15000      # define the maximum episodes
max_episode_test =1000
max_steps=50 #等于inc
max_steps_test=50
map_size = [300,300]     # maximum map size


# key parameters of birds
n_agents = 90  #
maxSpeed = 1     # L, maximum speed
communication_R=15
R_max=30#最大的通信半径
R_min=3
min_d=3
inc = 90
'''bird class'''
class Agents:
    def __init__(self):
        self.pos_old = np.array([random.uniform(-inc, inc),random.uniform(-inc, inc)])
        self.vel_old = np.array([0,0])
        self.pos_new = self.pos_old   # new position
        self.vel_new = self.vel_old   # new velocity
        self.done=0

        self.node1=n_agents
        self.node2=n_agents