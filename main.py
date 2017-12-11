from LinUCB import LinUCB
from LTS import LTS
from Statistic import Statistic
import Bagging
import Data
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
display,click,user_vec,pool=Data.load_data("data.txt")
data_size=len(display)
record=[]
agents = []
linucb=LinUCB(Data.USER_VEC_SIZE)
lts=LTS(Data.USER_VEC_SIZE)
stat = Statistic(Data.USER_VEC_SIZE)
agents.append(linucb)
agents.append(lts)
agents.append(stat)
hybrid = Bagging.HybridBagging(agents)
for i in range(data_size):
    pred=hybrid.predict_and_learn(user_vec[i],display[i],click[i],pool[i])
record = hybrid.reward_history
print ("Total valid events: {}".format(len(record)))
print ("Total Reward : {}".format(sum(record)))


