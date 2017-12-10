from Statistic import Statistic
import Data
display,click,user_vec,pool=Data.load_from_dump()
pool_size=len(pool)
data_size=len(display)
print("pool_size= {}, data_size={}".format(pool_size,data_size))
print("total clicks = {}".format(sum(click)))
#set up sta algo
sta=Statistic(pool_size,Data.USER_VEC_SIZE)
record=[]
for i in range(data_size):
    pred=sta.predict(user_vec[i])
    if pred == display[i]:
        record.append(click[i])

    sta.learn(user_vec[i],display[i],click[i])

print "Total valid events: {}".format(len(record))
print "Total Reward : {}".format(sum(record))

