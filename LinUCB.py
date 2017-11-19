import numpy as np
from numpy.random import choice
class LinUCB(object):
    def __init__(self,alpha,d,bandit_num):
        assert alpha >0
        assert d>0 and type(d) is int
        self.bandit_num=bandit_num
        self.alpha=alpha
        self.d=d
        self.A=np.eye(self.d)
        self.b=np.zeros((self.d,1))
        self.reward_track=[]
        
    
    def play(self,context,pull_results):
        
        assert context.shape==(self.bandit_num,self.d)
        theta=np.linalg.solve(self.A,self.b)
        value=np.zeros(self.bandit_num)
        for a in range(self.bandit_num): #for each action
            x=context[a].reshape(self.d,1)#x is a column vector
            ucb_squared=np.dot(x.T,np.linalg.solve(self.A,x))
            value[a]=np.dot(theta.T,x)+self.alpha*np.sqrt(ucb_squared)

        decision=choice(np.flatnonzero(value == value.max())) #break tie randomly
        reward=pull_results[decision]
        if len(self.reward_track) == 0:
            self.reward_track.append(reward)
        else:
            self.reward_track.append(reward+self.reward_track[-1])

        self.A=self.A+np.dot(context[decision],context[decision].T)
        self.b=self.b+reward*context[decision].reshape(self.d,1)
        

