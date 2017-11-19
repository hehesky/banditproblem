import random
import numpy as np

class BernoulliBandit(object):
    """Bernoulli Bandit, generate a positive reward R with probability of p
    constructor:
    BernoulliBandit(probability,reward)
    """
    def __init__(self,p,r=1):
        if p>1 or p<0:
            raise ValueError("Invalid probability {}".format(p))
        if r<=0:
            raise ValueError("Reward must be positive")
        self.probability=p
        self.reward=r
        self.action_value=r*p
    def pull(self):
        if random.random()<=self.probability:
            return self.reward
        else:
            return 0
        
class GaussianBandit(object):
    def __init__(self,mu=1,sigma=1):
        self.mu=mu
        self.sigma=sigma
        self.action_value=mu
    def pull(self):
        return random.gauss(self.mu,self.sigma)

class MultiArmedBandit(object):
    """A collection of bandits. Use an object of this class to feed into agents"""
    def __init__(self,bandits=None):
        self.optimal_bandit=None
        if bandits is None:
            self.bandits=[]
        else: 
            self.bandits=bandits
            
    def pull_all(self):
        """returns all rewards for one pull"""
        return [b.pull() for b in self.bandits ]
    
    def get_rewards(self, times):
        """will produce a N*m reward matrix where N is times and m is arms"""
        rewards = [self.pull_all() for i in range(times)]
        return np.asarray(rewards)


