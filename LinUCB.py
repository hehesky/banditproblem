import numpy as np
from numpy.random import choice
class LinUCB(object): #with disjoint linear models
    def __init__(self,feature_size,confidence=1):
        
        self.n=feature_size
        self.A={}
        self.b={}
        self.c=confidence
        self.total_runs = 0
        self.total_reward = 0

    def learn(self,context,aritcleID,reward):
        '''Update parameters with [context vector],[aritcleID(article chosen)],[reward(click or not)]'''
        assert aritcleID in self.A #for debug
        self.A[aritcleID]+=np.dot(context.reshape((self.n,1)),context.reshape(1,self.n))
        self.b[aritcleID]+=reward*context

    def predict(self,context,pool):
        '''Predict which article/bandit is most likely to generate a reward given a context vector(user traits)'''
        assert len(context)==self.n #for debug
        p=np.zeros(len(pool))
        for id in pool:
            if id not in self.A:
                self.A[id]=np.eye(self.n)
                self.b[id]=np.zeros(self.n)

            theta=np.linalg.solve(self.A[id],self.b[id])#theta=inv(A)*b
            confidence_bound=self.c*np.sqrt(np.dot(context,np.linalg.solve(self.A[id],context)))
            index,=np.where(pool==id)
            p[index]=np.dot(theta.T,context)+confidence_bound
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
    agent=LinUCB(feature_size=3,confidence=1)
    context=np.array([1,2,3])
    pool=[111,222,333]
    print(agent.predict(context,pool))
    agent.learn(context,111,1)
    agent.learn(context,222,0)
    agent.learn(context,111,0)
    agent.learn(context,333,0)
    print(agent.predict(context,pool))
