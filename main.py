from __future__ import print_function
import bandit
import agent

class BanditTest(object):
    def __init__(self,bandits,agents,n=1000):
        """bandits: MultiArmedBandit object, agents: a list of agents, n: number of rounds to play"""
        self.bandits=bandits
        self.agents=agents
        self.n=n
    def run(self,n=None):
        if n is not None:
            self.n=n

        for i in range(self.n):
            pull_result=self.bandits.pull_all()
            for agent in self.agents:
                agent.play(pull_result)

    def show_result(self):
        raise NotImplementedError


if __name__=="__main__":
    bandit_list=[]
    bandit_list.append(bandit.BernoulliBandit(0.5,1))
    bandit_list.append(bandit.GaussianBandit(1,0.5))
    bandit_num=len(bandit_list)
    KArmedBandit=bandit.MultiArmedBandit(bandit_list)

    agents=[agent.Agent(bandit_num)]

    test=BanditTest(KArmedBandit,agents,1000)
    test.run()

