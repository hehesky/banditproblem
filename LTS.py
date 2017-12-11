# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:27:12 2017

@author: sxy52
"""
import numpy as np
from numpy.random import choice
class LTS(object):        
    def __init__(self,feature_size):
        '''
        self.total_runs: the total times of making prediction
        self.total_reward: the total reward of prediction
        B, miu, f are median parameters
        '''
        self.n = feature_size
        self.B = {}
        self.miu = {}
        self.f = {}
        self.total_runs = 0
        self.total_reward = 0
    
    def learn(self, context, aritcleID, reward):
        assert aritcleID in self.B
        v = context.reshape((1,context.shape[0]))
        self.B[aritcleID] = self.B[aritcleID]+v.T.dot(v)
        self.f[aritcleID] = self.f[aritcleID] + v[0]*reward
        B_ite = np.linalg.solve(self.B[aritcleID],np.identity(self.n))
        self.miu[aritcleID] = B_ite.dot(self.f[aritcleID])
        
    def predict(self, context,pool):
        pool = np.asarray(pool)
        assert len(context)==self.n
        p=np.zeros(len(pool))
        v = context.reshape((1,context.shape[0]))
        for id in pool:
            if id not in self.B:
                self.B[id] = np.identity(self.n)
                self.f[id] = np.zeros(self.n)
                self.miu[id] = np.zeros(self.n)
                
            B_ite = np.linalg.solve(self.B[id],np.identity(self.n))
            miu_sample = np.random.multivariate_normal(self.miu[id], B_ite)
            index,=np.where(pool==id)
            p[index]=np.dot(v, miu_sample)
        decision=choice(np.flatnonzero(p == p.max()))
        return pool[decision]
        
    def accuracy_update(self, reward):
        '''
        update accuracy: after prediction, calculate new average reward
        '''
        self.total_runs += 1
        self.total_reward += reward
        return self.total_reward/self.total_runs
    
if __name__=='__main__':
    agent=LTS(feature_size=3)
    context=np.array([1,2,3])
    pool=[111,222,333]
    print(agent.predict(context,pool))
    agent.learn(context,111,1)
    agent.learn(context,222,0)
    agent.learn(context,111,0)
    agent.learn(context,333,0)
    print(agent.predict(context,pool))