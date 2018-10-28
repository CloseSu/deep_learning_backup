from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples

        # if self.num_experiences < batch_size:
        #     return random.sample(self.buffer, self.num_experiences)
        # else:
        #     return random.sample(self.buffer, batch_size)

        states, actions, rewards, new_states, dones = [], [], [], [], []
        batch = random.sample(self.buffer, self.num_experiences if self.num_experiences < batch_size else batch_size)
        for data in batch:
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            new_states.append(data[3])
            dones.append(data[4])
        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(dones)




    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
