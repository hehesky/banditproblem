from LinUCB import LinUCB
from LTS import LTS
from Statistic import Statistic

from numpy.random import choice
import numpy as np
import Bagging
import Data
import plot
import time

class Sep_test(object):
    def __init__(self,feature_size):
        '''
        self.total_runs: the total times of making prediction
        self.total_reward: the total reward of prediction
        B, miu, f are median parameters
        '''
        self.linucb = LinUCB(feature_size)
        self.lts = LTS(feature_size)
        self.stat = Statistic(feature_size)
        self.his_linucb = []
        self.his_lts = []
        self.his_stat = []
        self.his_hybrid = []
        self.valid_linucb = 0
        self.valid_lts = 0
        self.valid_stat = 0
        self.valid_hybrid = 0
        self.vote = []
        
    def LinUCB_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.linucb.predict(context,pool)
        self.vote.append(prediction)
        #update records
        if prediction==articleID:
            self.his_linucb.append(reward)
            self.valid_linucb += 1
        #train one of the agents
        self.linucb.learn(context,articleID,reward)
        
    def lts_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.lts.predict(context,pool)
        self.vote.append(prediction)
        #update records
        if prediction==articleID:
            self.his_lts.append(reward)
            self.valid_lts += 1
        #train one of the agents
        self.lts.learn(context,articleID,reward)
        
    def stat_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.stat.predict(context,pool)
        self.vote.append(prediction)
        #update records
        if prediction==articleID:
            self.his_stat.append(reward)
            self.valid_stat += 1
        #train one of the agents
        self.stat.learn(context,articleID,reward)
        
    def predict_and_learn(self,context,articleID,reward,pool):
        self.LinUCB_predict_and_learn(context,articleID,reward,pool)
        self.lts_predict_and_learn(context,articleID,reward,pool)
        self.stat_predict_and_learn(context,articleID,reward,pool)
        
    def Hybrid_predict_and_learn(self,context,articleID,reward,pool):
        self.predict_and_learn(context,articleID,reward,pool)
        counts=np.bincount(self.vote)
        prediction=choice(np.flatnonzero(counts == counts.max()))

        #update records
        if prediction==articleID:
            self.his_hybrid.append(reward)
            self.valid_hybrid+=1
            
        self.vote=[]

print("==START==")
start_time = time.time()
data_dir = 'rewrite.txt'
batch_num = Data.process_large_data(data_dir)
data_gen=Data.get_batched_data(min(batch_num,3))
print("done processing data file")

seprate_test = Sep_test(Data.USER_VEC_SIZE)
print("Computation starts")

total_click=0
total_data=0#count data entries
for (display,click,user_vec,pool) in data_gen:
    #do something with current data
    total_data+=1
    total_click+=click
    seprate_test.Hybrid_predict_and_learn(user_vec,display,click,pool)

total_crt=total_click*1.0/total_data
print(total_crt)
record_linucb = seprate_test.his_linucb
record_lts = seprate_test.his_lts
record_stat = seprate_test.his_stat
record_hybrid=seprate_test.his_hybrid
print("Done computation")


avg_hybrid=plot.cumulative_avg(record_hybrid)/total_crt
np.savetxt("hybrid.csv",avg_hybrid,delimiter=',')
avg_linucb=plot.cumulative_avg(record_linucb)/total_crt
np.savetxt("linucb.csv",avg_linucb,delimiter=',')
avg_lts=plot.cumulative_avg(record_lts)/total_crt
np.savetxt('lts.csv',avg_lts,delimiter=',')
avg_stat=plot.cumulative_avg(record_stat)/total_crt
np.savetxt('stat.csv',avg_stat,delimiter=',')

plot.plot_avg(
    (avg_hybrid,avg_linucb,avg_lts,avg_stat),
    title="Average Reward",
    filename='plot.png',
    legend=['hybrid','linucb','lts','stat'],
    xlabel="Sample Size",
    ylabel="Average Reward")

end_time=time.time()
time_used=end_time-start_time
print("Total time used: {}".format(time_used))