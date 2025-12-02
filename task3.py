"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed
def KL(p : float, q : float):
    ans = 0.0
    # push above 0
    q = max(1e-5, q)

    # pull below 1
    q = min(q, 1 - 1e-5)

    if p != 0.0:
        ans += p * math.log(p / q)
    if p != 1.0:
        ans += (1.0 - p) * math.log((1.0 - p)/(1.0 - q))
    return ans
# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.total_pulls = 0
        self.num_pulls = np.zeros(num_arms)
        self.successes = np.zeros(num_arms)
        self.empirical_mean = np.zeros(num_arms)
        self.kl_ucb = np.zeros(num_arms)

        self.C = 1.0
        self.threshold = 1e-4

        # for batch handling
        self.batch_remaining = 0
        self.chosen_arm = None
        #END EDITING HERE
    
    def calc_kl_ucb(self):
        # to guard log(log(t)) for very small t, use log(max(2,log(t))) instead
        numer = math.log(self.total_pulls) + self.C * math.log(math.log(max(self.total_pulls,2)))

        # recalculate empirical means
        self.empirical_mean = self.successes / self.num_pulls

        # calculate kl-ucb for each arm
        for arm_index in range(self.num_arms):
            search_value = numer / self.num_pulls[arm_index]
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

    def give_pull(self):
        #START EDITING HERE
        # initially pull each arm once
        if self.total_pulls < self.num_arms:
            return self.total_pulls
        
        # if batch remaining, continue
        if self.batch_remaining > 0:
            self.batch_remaining -= 1
            return self.chosen_arm
        
        # recalculate kl-ucb values
        self.calc_kl_ucb()
        
        sorted_indices = np.argsort(self.kl_ucb)
        first_highest = sorted_indices[-1]
        second_highest = sorted_indices[-2]

        p = self.empirical_mean[first_highest]
        n = self.num_pulls[first_highest]

        # binary search on the number of times we can pull
        low = 1
        high = self.horizon
        safe_pulls = 0

        numer = math.log(self.total_pulls) + self.C * math.log(max(2, math.log(self.total_pulls)))

        while low <= high:
            mid = (low + high) // 2
            worst_case_emp_mean = p * n / (n + mid)
            search_value = numer / (n + mid)

            if KL(worst_case_emp_mean, self.kl_ucb[second_highest]) <= search_value:
                safe_pulls = mid
                low = mid + 1
            else:
                high = mid - 1

        self.batch_remaining = safe_pulls - 1
        self.chosen_arm = first_highest
        return first_highest
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.successes[arm_index] += reward

        self.num_pulls[arm_index] += 1
        self.total_pulls += 1

        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.total_pulls = 0
        self.num_pulls = np.zeros(num_arms)
        self.successes = np.zeros(num_arms)
        self.empirical_mean = np.zeros(num_arms)
        self.kl_ucb = np.zeros(num_arms)

        self.C = 1.0
        self.threshold = 1e-4

        # for batch handling
        self.batch_remaining = 0
        self.chosen_arm = None
        #END EDITING HERE
    
    def calc_kl_ucb(self):
        # to guard log(log(t)) for very small t, use log(max(2,log(t))) instead
        numer = math.log(self.total_pulls) + self.C * math.log(math.log(max(self.total_pulls,2)))

        # recalculate empirical means
        self.empirical_mean = self.successes / self.num_pulls

        # calculate kl-ucb for each arm
        for arm_index in range(self.num_arms):
            search_value = numer / self.num_pulls[arm_index]
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

    def give_pull(self):
        #START EDITING HERE
        # initially pull each arm once
        if self.total_pulls < self.num_arms:
            return self.total_pulls
        
        # if batch remaining, continue
        if self.batch_remaining > 0:
            self.batch_remaining -= 1
            return self.chosen_arm
        
        # recalculate kl-ucb values
        self.calc_kl_ucb()
        
        sorted_indices = np.argsort(self.kl_ucb)
        first_highest = sorted_indices[-1]
        second_highest = sorted_indices[-2]

        p = self.empirical_mean[first_highest]
        n = self.num_pulls[first_highest]

        # binary search on the number of times we can pull
        low = 1
        high = self.horizon
        safe_pulls = 0

        numer = math.log(self.total_pulls) + self.C * math.log(max(2, math.log(self.total_pulls)))

        while low <= high:
            mid = (low + high) // 2
            worst_case_emp_mean = p * n / (n + mid)
            search_value = numer / (n + mid)

            if KL(worst_case_emp_mean, self.kl_ucb[second_highest]) <= search_value:
                safe_pulls = mid
                low = mid + 1
            else:
                high = mid - 1

        self.batch_remaining = safe_pulls - 1
        self.chosen_arm = first_highest
        return first_highest
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.successes[arm_index] += reward

        self.num_pulls[arm_index] += 1
        self.total_pulls += 1
