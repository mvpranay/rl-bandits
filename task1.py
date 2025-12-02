"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(p : float, q : float):
    ans = 0.0
    # push above 0
    q = max(q, 1e-5)
    # pull below 1
    q = min(q, 1 - 1e-5)
    if p != 0.0:
        ans += p * math.log(p / q)
    if p != 1.0:
        ans += (1.0 - p) * math.log((1.0 - p)/(1.0 - q))
    return ans
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.horizon = horizon
        self.total_pulls = 0
        self.num_pulls = np.zeros(num_arms)
        self.successes = np.zeros(num_arms)
        self.empirical_mean = np.ones(num_arms)
        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START HERE
        # first pull each arm once
        if self.total_pulls < self.num_arms:
            return self.total_pulls
        
        # empirical_mean + exploration_bonus
        exploration_bonus = np.sqrt(1.5 * math.log(self.total_pulls) / self.num_pulls)
        self.ucb = self.empirical_mean + exploration_bonus
        return np.argmax(self.ucb)
        
        # END EDITING HERE  
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.successes[arm_index] += reward
        self.num_pulls[arm_index] += 1
        self.total_pulls += 1

        # update empirical mean for this arm
        self.empirical_mean[arm_index] = self.successes[arm_index] / self.num_pulls[arm_index]
        
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.total_pulls = 0
        self.num_pulls = np.zeros(num_arms)
        self.successes = np.zeros(num_arms)
        self.empirical_mean = np.zeros(num_arms)
        self.kl_ucb = np.zeros(num_arms)

        self.C = 0
        self.threshold = 1e-6
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        # initially pull each arm once
        if self.total_pulls < self.num_arms:
            arm_index = self.total_pulls
            return arm_index
        
        # calculate kl-ucb for each arm
        for arm_index in range(self.num_arms):
            search_value = math.log(self.total_pulls) + self.C * math.log(math.log(max(self.total_pulls,2)))
            search_value /= self.num_pulls[arm_index]

            # find max q such that KL(p,q) <= search_value
            p = self.empirical_mean[arm_index]
            low = p
            high = 1.0
            q = low

            # binary search
            while (low + self.threshold < high):
                mid = (low + high) / 2.0
                if KL(p, mid) <= search_value:
                    q = mid
                    low = mid
                else:
                    high = mid

            self.kl_ucb[arm_index] = q

        return np.argmax(self.kl_ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.successes[arm_index] += reward

        self.num_pulls[arm_index] += 1
        self.total_pulls += 1

        # update empirical mean for this arm
        self.empirical_mean[arm_index] = self.successes[arm_index] / self.num_pulls[arm_index]
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.horizon = horizon
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # sample from beta distribution for each arm
        value = np.random.beta(self.successes + 1.0, self.failures + 1.0)
        return np.argmax(value)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 0:
            self.failures[arm_index] += 1
        else:
            self.successes[arm_index] += 1
        # END EDITING HERE

 
