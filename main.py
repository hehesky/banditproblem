from LinUCB import LinUCB
from LTS import LTS
from Statistic import Statistic
import Bagging
import Data

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
        self.stat = 0
        
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
'''
display,click,user_vec,pool=Data.load_from_dump()
data_size=len(display)
print("data_size={}".format(data_size))
print("total clicks = {}".format(sum(click)))
#set up algo
linucb = LinUCB(Data.USER_VEC_SIZE)
record=[]
for i in range(data_size):
    pred=linucb.predict(user_vec[i],pool[i])
    if pred==display[i]:
        record.append(click[i])
    linucb.learn(user_vec[i],display[i],click[i])

print ("Total valid events: {}".format(len(record)))
print ("Total Reward : {}".format(sum(record)))
'''
data_dir = ''
batch_num = Data.process_large_data(data_dir)
data_gen=Data.get_batched_data(batch_num)

record=[]
agents = []
linucb=LinUCB(Data.USER_VEC_SIZE)
lts=LTS(Data.USER_VEC_SIZE)
stat = Statistic(Data.USER_VEC_SIZE)
agents.append(linucb)
agents.append(lts)
agents.append(stat)
hybrid = Bagging.HybridBagging(agents)
seprate_test = Sep_test(Data.USER_VEC_SIZE)
for (display,click,user_vec,pool) in data_gen:
    #do something with current data
    hybrid.predict_and_learn(user_vec,display,click,pool)
    seprate_test.predict_and_learn(user_vec,display,click,pool)
record = hybrid.reward_history
record_linucb = seprate_test.his_linucb
record_lts = seprate_test.his_lts
record_stat = seprate_test.his_stat

print ("Total valid events: {}".format(len(record)))
print ("Total Reward : {}".format(sum(record)))


