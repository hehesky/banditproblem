# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 08:31:21 2017

@author: sxy52
"""
import numpy as np
import bandit

class Greedy(object):
    '''
    this is the epislon greedy algrithm for bandit problem
    '''
    def __init__(self, k = 2, epislon = 0.3):
        """
        optimal: the arm with the highest mean
        k: amount of arms
        total_reward: an analysis of performance
        actions: times of pulling of every arm
        mius: means of every available arm 
        
        *epislon: parameter for epislon greedy
        """
        self.optimal = 0
        self.k = k
        self.total_reward = 0
        self.mius = np.zeros(k)
        
        self.epislon = epislon       

    def training(self, rewards, times):
        '''
        testing execution
        '''
        for j in range(self.k):
            self.mius[j] = np.sum([rewards[i*self.k+j,j] for i in range(times)])
            self.total_reward += self.mius[j]
        self.mius = self.mius/times
        return self.total_reward
    
    def play(self, rewards):
        train_times = int(rewards.shape[0]*self.epislon/self.k)
        test_times = rewards.shape[0] - train_times*self.k
        self.training(rewards, train_times)
        
        test_rewards = rewards[-test_times:,:]
        print(train_times)
        print(test_rewards.shape)
        
        self.optimal = np.argmax(self.mius)
        self.total_reward += np.sum(test_rewards[:,self.optimal])
        return self.total_reward, self.optimal

def bandit_test():
    """
    test sample: 
        initial a multi-armed bandit and get a reward matrix
        test play
    """
    #initial multi-armed bandit
    bandits = []
    arms = 11
    for i in range(10):
        berbandit = bandit.BernoulliBandit(0.5,1)
        bandits.append(berbandit)
    berbandit = bandit.BernoulliBandit(0.8,1)
    bandits.append(berbandit)    
    
    bandit_test = bandit.MultiArmedBandit(bandits)
    #get reward matrix for 100 plays
    rewards = bandit_test.get_rewards(10000)
    print(rewards.shape)
    #test one play
    test = Greedy(arms, 0.3)
    print(test.play(rewards))
    return 

if __name__=="__main__":
    bandit_test()