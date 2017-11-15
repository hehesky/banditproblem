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
        optimal: the arm with the highest mean
        k: amount of arms
        total_reward: an analysis of performance
        actions: times of pulling of every arm
        mius: means of every available arm 
        bandit: the multiarmed bandit machine
        """
        self.optimal = 0
        self.k = k
        self.total_reward = 0
        self.actions = np.zeros(k)
        self.mius = np.zeros(k)
        self.bandit = bandit     
        
        self.epislon = epislon       

    def one_play(self):
        '''
        one execution
        '''
        num = np.random.random()
        if(num < self.epislon):
            exe = self.optimal
        else:
            exe = np.random.randint(self.k)
        #pull and get reward
        reward = self.bandit.pull(exe)
        self.total_reward += reward
        #update means
        self.mius[exe] = self.mius[exe]*self.actions[exe]
        self.actions[exe] += 1
        self.mius[exe] = (self.mius[exe]+reward)/self.actions[exe]
        self.optimal = np.argmax(self.mius)
        return self.total_reward
    
    def play(self, times):
        if times <= 0:
            raise ValueError
        else:
            for i in range(times):
                self.one_play()
            print(self.actions)
            return self.total_reward, np.argmax(self.mius)
                
def bandit_test():
    bandits = []
    arms = 11
    for i in range(10):
        berbandit = bandit.BernoulliBandit(0.5,1)
        bandits.append(berbandit)
    berbandit = bandit.BernoulliBandit(0.8,1)
    bandits.append(berbandit)
    bandit_test = bandit.MultiArmedBandit(bandits)
    test = EpiGreedy(bandit_test, arms, 0.1)
    print(test.play(100))
    return 

if __name__=="__main__":
    bandit_test()


