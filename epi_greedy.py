# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:17:31 2017

@author: sxy52
references:
    https://blog.thedataincubator.com/2016/07/multi-armed-bandits-2/
"""
import numpy as np
import bandit

class EpiGreedy(object):
    '''
    this is the epislon greedy algrithm for bandit problem
    '''
    def __init__(self, bandit, k = 2, epislon = 1):
        """
        *epislon: parameter for epislon greedy
        k: amount of arms
        i: the i-th arm with the highest mean
        action: times of pulling of every arm
        mius: means of every available arm 
        """
        self.i = 0
        self.k = k
        self.total_reward = 0
        self.actions = np.zeros(k)
        self.bandit = bandit        
        self.epislon = epislon       
        self.mius = np.zeros(k)
        
    def one_rand(self):
        '''
        one execution
        '''
        num = np.random.random()
        if(num > self.epislon):
            exe = np.argmax(self.mius)
        else:
            exe = np.random.randint(self.k)
        #get reward adn update data after pull
        exe = self.one_rand()
        reward = self.bandit.pull()
        self.total_reward += reward
        self.mius[exe] = self.mius[exe]*self.actions[exe]
        self.actions[exe] += 1
        self.mius = (self.mius[exe]+reward)/self.actions[exe]
        return self.total_reward
    
    def multi_rand(self, times = 1):
        if times <= 0:
            raise NotImplementedError
        else:
            for i in range(times):
                self.one_rand()
            return self.total_reward, np.argmax(self.mius)
                
def bandit_test():
    bandit_test = bandit.MultiArmedBandit()
    return 

if __name__=="__main__":
    bandit_test()


