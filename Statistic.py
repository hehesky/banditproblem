# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:36:02 2017

@author: sxy52
"""
import numpy as np
from numpy.random import choice
class Statistic(object):
    def __init__(self,feature_size):
        self.ft_num = feature_size
        self.record = {}
        self.ft_count = np.ones(self.ft_num)
        self.arm_count = {}
        self.count = 2
        self.total_runs = 0
        self.total_reward = 0
        
        
    def learn(self,ft_vec, aritcleID, click):
        ft_vec = np.asarray(ft_vec)
        self.record[aritcleID] += ft_vec*click
        self.ft_count += ft_vec
        self.arm_count[aritcleID] += click
        self.count += 1
        
    def predict(self, ft_vec,pool):
        pool=np.asarray(pool)
        assert len(ft_vec)==self.ft_num
        prob=np.zeros(len(pool))
        ft_vec = np.asarray(ft_vec)
        for id in pool:
            if id not in self.arm_count:
                self.arm_count[id]=1
                self.record[id] = np.ones(self.ft_num)
                
            condition = ft_vec==1
            n_count = (self.ft_count[condition == False] - self.count) * (-1)
            n_prob = (self.record[id][condition == False] - self.arm_count[id])*(-1)
            n_cond_p = np.prod(n_prob/n_count)
            index,=np.where(pool==id)
            prob[index] = np.prod(self.record[id][condition]/self.ft_count[condition])*n_cond_p
        decision=choice(np.flatnonzero(prob == prob.max()))
        return pool[decision]
    
    def accuracy_update(self, reward):
        '''
        update accuracy: after prediction, calculate new average reward
        '''
        self.total_runs += 1
        self.total_reward += reward
        return self.total_reward/self.total_runs

if __name__=='__main__':
    agent=Statistic(feature_size=3)
    context=np.array([1,0,1])
    pool=[111,222,333]
    print(agent.predict(context,pool))
    agent.learn(context,111,1)
    agent.learn(context,222,0)
    agent.learn(context,111,0)
    agent.learn(context,333,0)
    print(agent.predict(context,pool))
    
    