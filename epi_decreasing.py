# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:17:31 2017

@author: sxy52
references:
    https://blog.thedataincubator.com/2016/07/multi-armed-bandits-2/
"""
import numpy as np
import bandit

class EpiDecreasing(object):
    '''
    this is the epislon greedy algrithm for bandit problem
    '''
    def __init__(self, k = 2):
        """
        optimal: the arm with the highest mean
        k: amount of arms
        total_reward: an analysis of performance
        actions: times of pulling of every arm
        mius: means of every available arm 
        """
        self.optimal = 0
        self.k = k
        self.total_reward = 0
        self.actions = np.zeros(k)
        self.mius = np.zeros(k)   
        
    def training(self, rewards, times):
        '''
        testing execution
        '''
        for j in range(self.k):
            self.mius[j] = np.sum([rewards[i*self.k+j,j] for i in range(times)])
            self.total_reward += self.mius[j]
        self.mius = self.mius/times
        self.optimal = np.argmax(self.mius)
        return self.total_reward

    def one_play(self, rewards, epsilon):
        '''
        one execution
        '''
        num = np.random.random()
        if(num > epsilon):
            exe = self.optimal
        else:
            exe = np.random.randint(self.k)
        #pull and get reward
        reward = rewards[exe]
        self.total_reward += reward
        #update means
        self.mius[exe] = self.mius[exe]*self.actions[exe]
        self.actions[exe] += 1
        self.mius[exe] = (self.mius[exe]+reward)/self.actions[exe]
        self.optimal = np.argmax(self.mius)
        return self.total_reward
    
    def play(self, rewards):  
        #training and get initial optimal
        self.training(rewards, 10)
        #epsilon decreasing
        current = 10*self.k
        test_times = rewards.shape[0]-current
        test_rewards = rewards[current:,:]
        
        epsilon = 1
        for i in range(test_times):
            #for each play, input rewards for one pull
            self.one_play(test_rewards[i],epsilon)
            epsilon = 1/(i+1)
        print(self.actions)
        return self.total_reward, np.argmax(self.mius)
                
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
    test = EpiDecreasing(arms)
    print(test.play(rewards))
    return 

if __name__=="__main__":
    bandit_test()


