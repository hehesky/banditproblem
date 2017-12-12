from LinUCB import LinUCB
from LTS import LTS
from Statistic import Statistic
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
        self.valid_linucb = 0
        self.valid_lts = 0
        self.valid_stat = 0
        
    def LinUCB_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.linucb.predict(context,pool)
        #update records
        if prediction==articleID:
            self.his_linucb.append(reward)
            self.valid_linucb += 1
        #train one of the agents
        self.linucb.learn(context,articleID,reward)
        
    def lts_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.lts.predict(context,pool)
        #update records
        if prediction==articleID:
            self.his_lts.append(reward)
            self.valid_lts += 1
        #train one of the agents
        self.lts.learn(context,articleID,reward)
        
    def stat_predict_and_learn(self,context,articleID,reward,pool):
        prediction = self.stat.predict(context,pool)
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

print("==START==")
start_time = time.time()
data_dir = 'ydata-fp-td-clicks-v2_0.20111003'#'rewrite.txt'
batch_num = Data.process_large_data(data_dir)
data_gen=Data.get_batched_data(min(batch_num,3))
print("done processing data file")

agents = []
linucb=LinUCB(Data.USER_VEC_SIZE)
lts=LTS(Data.USER_VEC_SIZE)
stat = Statistic(Data.USER_VEC_SIZE)
agents.append(linucb)
agents.append(lts)
agents.append(stat)
hybrid = Bagging.HybridBagging(agents)

agents=[LinUCB(Data.USER_VEC_SIZE) for i in range(3)]+[LTS(Data.USER_VEC_SIZE) for i in range(3)]+[Statistic(Data.USER_VEC_SIZE) for i in range(3)]
bag=Bagging.Bagging(agents)
seprate_test = Sep_test(Data.USER_VEC_SIZE)
print("Computation starts")

total_click=0
total_data=0#count data entries
for (display,click,user_vec,pool) in data_gen:
    #do something with current data
    total_data+=1
    total_click+=click
    bag.predict_and_learn(user_vec,display,click,pool)
    hybrid.predict_and_learn(user_vec,display,click,pool)
    seprate_test.predict_and_learn(user_vec,display,click,pool)

total_crt=total_click*1.0/total_data
print(total_crt)
record_hybrid = hybrid.reward_history
record_linucb = seprate_test.his_linucb
record_lts = seprate_test.his_lts
record_stat = seprate_test.his_stat
record_bag=bag.reward_history
print("Done computation")


avg_hybrid=plot.cumulative_avg(record_hybrid)/total_crt
np.savetxt("hybrid.csv",avg_hybrid,delimiter=',')
avg_linucb=plot.cumulative_avg(record_linucb)/total_crt
np.savetxt("linucb.csv",avg_linucb,delimiter=',')
avg_lts=plot.cumulative_avg(record_lts)/total_crt
np.savetxt('lts.csv',avg_lts,delimiter=',')
avg_stat=plot.cumulative_avg(record_stat)/total_crt
np.savetxt('stat.csv',avg_stat,delimiter=',')
avg_bag=plot.cumulative_avg(record_bag)/total_crt

plot.plot_avg(
    (avg_hybrid,avg_linucb,avg_lts,avg_stat,avg_bag),
    title="Average Reward",
    filename='plot.png',
    legend=['hybrid','linucb','lts','stat','bagging'],
    xlabel="Sample Size",
    ylabel="Average Reward")

end_time=time.time()
time_used=end_time-start_time
print("Total time used: {}".format(time_used))