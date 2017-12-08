# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:36:02 2017

@author: sxy52
"""
import numpy as np
from numpy.random import choice
class Statistic(object):
    def __init__(self,bandit_num,feature_num):
        self.ft_num = feature_num
        self.arm_num = bandit_num
        self.record = np.ones((self.arm_num, self.ft_num))
        self.ft_count = np.ones(self.ft_num)
        self.arm_count = np.ones(self.arm_num)
        self.count = 2
        self.total_runs = 0
        self.total_reward = 0
        
    def learn(self,ft_vec, arm, click):
        ft_vec = np.asarray(ft_vec)
        self.record[arm] += ft_vec*click
        self.ft_count += ft_vec
        self.arm_count[arm] += click
        self.count += 1
        
    def predict(self, ft_vec):
        ft_vec = np.asarray(ft_vec)
        prob = np.zeros(self.arm_num)
        for i in range(self.arm_num):
            condition = ft_vec==1
            n_count = (self.ft_count[condition == False] - self.count) * (-1)
            n_prob = (self.record[i][condition == False] - self.arm_count[i])*(-1)
            n_cond_p = np.prod(n_prob/n_count)
            prob[i] = np.prod(self.record[i][condition]/self.ft_count[condition])*n_cond_p
        decision=choice(np.flatnonzero(prob == prob.max()))
        return decision
    
    def accuracy_update(self, reward):
        '''
        update accuracy: after prediction, calculate new average reward
        '''
        self.total_runs += 1
        self.total_reward += reward
        return self.total_reward/self.total_runs

if __name__=='__main__':
    agent=Statistic(2,3)
    context=np.array([1,2,3])
    agent.learn(context,1,1)
    agent.learn(context,0,0)
    agent.learn(context,1,0)
    print(agent.predict(context))
    
    