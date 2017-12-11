
try:
    import tkinter
except ImportError:

    import matplotlib
    matplotlib.use('agg')
finally:
    import matplotlib.pyplot as PLT
import numpy as np
def cumulative_avg(reward_record):
    cum_sum=np.cumsum(reward_record)
    indices=np.array(range(1,len(reward_record)+1))
    cum_avg=cum_sum*1.0/indices
    return cum_avg

def plot_avg(cum_avgs,title="Plot",filename=None,legend=None,xlabel=None,ylabel=None):
    
    PLT.figure()
    for avg in cum_avgs:
        PLT.plot(avg)
    PLT.title(title)
    if legend is not None:
        PLT.legend(legend)
    if xlabel is not None:
        PLT.xlabel(xlabel)
    if ylabel is not None:
        PLT.ylabel(ylabel)
    if filename is None:
        filename=title+'.png'
    PLT.savefig(filename)

record=[0,1,0,1,0,0,1]
avg=cumulative_avg(record)
avg2=cumulative_avg([1,1,1,1,1,0,1])

plot_avg([avg,avg2],"plot 1",legend=['line 1','line 2'],xlabel="x",ylabel='y')
