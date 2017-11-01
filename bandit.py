
# coding: utf-8

# In[7]:


import random
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
    def pull(self):
        if random.random()<=self.probability:
            return self.reward
        else:
            return 0
        
class GaussianBandit(object):
    def __init__(self,mu=1,sigma=1):
        self.mu=mu
        self.sigma=sigma
    
    def pull(self):
        return random.gauss(self.mu,self.sigma)
    
b1=BernoulliBandit(0.5,2)
b2=GaussianBandit(2,1)
print b1.pull()
print b2.pull()


# In[3]:




