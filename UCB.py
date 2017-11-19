from __future__ import division
from agent import Agent
import numpy as np

INF = float('inf')

class UCBAgent(Agent):
    """Upper Confidence Bound Agent
    Inherit from base Agent class of agent.py
    Implements self.play() interface

    Algorithm: For each bandit/agent A[i], compute a bound of confidence ucb[i],
    ucb[i]=sqrt((2*ln(t))/N(A[i])),
    where t is the number of rounds played and N(A[i]) is the number of times
    action A[i] used/taken.

    During each round, the agent choose the action with highes sum of ucb and average reward.

    """
    def __init__(self, n):
        Agent.__init__(self, n)

    def play(self, pull_results):
        value = []

        #list of decision value (i.e. the sum of confidence bound and current avg reward)
        for i in range(self.bandit_num):
            pull_count = len(self.pull_history[i])
            
            if pull_count == 0:
                avg_reward = 0 #set avg reward to 0 if a bandit is not played at all
            else:
                avg_reward = np.average(self.pull_history[i])
            if pull_count == 0:
                ucb = INF# set ucb to infinity if a bandit is not played at all
            else:
                ucb = np.sqrt(2*np.math.log(self.round)/pull_count)
            value.append(ucb+avg_reward)
        decision = value.index(max(value))
        #pull the bandit
        reward = pull_results[decision]

        if self.round == 0:
            self.reward_track=[reward]
        else:
            self.reward_track.append(reward+self.reward_track[-1])

        self.pull_history[decision]=np.append(self.pull_history[decision],reward)
        self.round += 1
