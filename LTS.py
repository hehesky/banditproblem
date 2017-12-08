# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:27:12 2017

@author: sxy52
"""
import numpy as np
from numpy.random import choice
class LTS(object):        
    def __init__(self, bandit_num,feature_size):
        '''
        self.total_runs: the total times of making prediction
        self.total_reward: the total reward of prediction
        B, miu, f are median parameters
        '''
        self.d = bandit_num
        self.n = feature_size
        self.B = [np.identity(self.n) for i in range(self.d)]
        self.miu = [np.zeros(self.n) for i in range(self.d)]
        self.f = [np.zeros(self.n) for i in range(self.d)]
        self.total_runs = 0
        self.total_reward = 0
    
    def learn(self, context, decision, reward):
        v = context.reshape((1,context.shape[0]))
        self.B[decision] = self.B[decision]+v.T.dot(v)
        self.f[decision] = self.f[decision] + v[0]*reward
        B_ite = np.linalg.solve(self.B[decision],np.identity(self.n))
        self.miu[decision] = B_ite.dot(self.f[decision])
        
    def predict(self, context):
        assert len(context)==self.n
        v = context.reshape((1,context.shape[0]))
        p=np.zeros(self.d)
        for i in range(self.d):
            B_ite = np.linalg.solve(self.B[i],np.identity(self.n))
            miu_sample = np.random.multivariate_normal(self.miu[i], B_ite)
            p[i]=np.dot(v, miu_sample)
        decision=choice(np.flatnonzero(p == p.max()))
        return decision
    
    def accuracy_update(self, reward):
        '''
        update accuracy: after prediction, calculate new average reward
        '''
        self.total_runs += 1
        self.total_reward += reward
        return self.total_reward/self.total_runs
    
if __name__=='__main__':
    agent=LTS(2,3)
    context=np.array([1,2,3])
    agent.learn(context,1,1)
    agent.learn(context,1,0)
    agent.learn(context,1,1)
    first = 0
    second = 0
    for i in range(1000):
        if(agent.predict(context)==0):
            first += 1
        else:
            second += 1
    print(first,second)