import numpy as np
from numpy.random import choice
class LinUCB(object): #with disjoint linear models
    def __init__(self,bandit_num,feature_size,confidence=1):
        self.d=bandit_num
        self.n=feature_size
        self.A=[np.eye(self.n) for i in range(self.d)]
        self.b=[np.zeros(self.n) for i in range(self.d)]
        self.c=confidence
        self.total_runs = 0
        self.total_reward = 0

    def learn(self,context,aritcleID,reward):
        '''Update parameters with [context vector],[aritcleID(article chosen)],[reward(click or not)]'''
        self.A[aritcleID]+=np.dot(context.reshape((self.n,1)),context.reshape(1,self.n))
        self.b[aritcleID]+=reward*context

    def predict(self,context):
        '''Predict which article/bandit is most likely to generate a reward given a context vector(user traits)'''
        assert len(context)==self.n #for debug
        p=np.zeros(self.d)
        for i in range(self.d):
            
            theta=np.linalg.solve(self.A[i],self.b[i])#theta=inv(A)*b
            confidence_bound=self.c*np.sqrt(np.dot(context,np.linalg.solve(self.A[i],context)))
            p[i]=np.dot(theta.T,context)+confidence_bound
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
    agent=LinUCB(2,3,1)
    context=np.array([1,2,3])
    agent.learn(context,1,1)
    agent.learn(context,0,0)
    agent.learn(context,1,0)
    print(agent.predict(context))
