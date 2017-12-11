from LinUCB import LinUCB
import Data
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

