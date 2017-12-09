import numpy as np
from numpy.random import choice
class Bagging(object):
    def __init__(self,agents):
        self.agents=agents
        self.agent_num=len(agents)
        self.cur_agent=0
        self.valid_entry=0
        self.reward_history=[]

    def predict_and_learn(self,context,articleID,reward):
        '''
        (1darray, int, int) -> None
        @param
        context -> [1darray] vector representing user feature
        aritcleID -> [int] index of article that are shown to user
        reward -> [int] 0 or 1 based on whether the user clicked on the article

        @description:
        All agents within votes on article to display, then the article with most votes are elected.
        Tie breaks randomly
        Records on reward history is updated when the event is valid (i.e. prediction == aritcleID)

        Then one of the agents is trained, round-robin style
        '''
        #all agents cast votes based on context 
        votes=np.array([agent.predict(context) for agent in self.agents])
        counts=np.bincount(votes)
        prediction=choice(np.flatnonzero(counts == counts.max()))

        #update records
        if prediction==articleID:
            self.reward_history.append(reward)
            self.valid_entry+=1

        #train one of the agents
        self.train(context,articleID,reward)
   
    def train(self, context, articleID, reward):
        '''
        (1darray, int, int) -> None
        @param
        context -> [1darray] vector representing user feature
        aritcleID -> [int] index of article that are shown to user
        reward -> [int] 0 or 1 based on whether the user clicked on the article

        Train one of the agents
        '''
        self.agents[self.cur_agent].learn(context,articleID,reward)
        self.cur_agent+=1
        if self.cur_agent>=self.agent_num:
            self.cur_agent=0
