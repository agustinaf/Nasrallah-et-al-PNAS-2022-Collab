# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:57:25 2022

@author: M. Agustina Frechou
"""

'''
IMPORT PACKAGES
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import os

'''
FUNCTIONS
'''
def filter1(traces, window_size=100):
   
    traces_filtered = []
    for ii in traces: 
        # Initialize an empty list to store moving averages
        i = 0
        moving_averages = []
        # Loop through the array to consider
        # every window of size 3
        while i < len(ii) - window_size + 1:
            
            # Store elements from i to i+window_size
            # in list to get the current window
            window = ii[i : i + window_size]
          
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
              
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
              
            # Shift window to right by one position
            i += 1
          
        traces_filtered.append(moving_averages)
    return traces_filtered

def activity(traces, start, finish):

    activity = []
    for i in traces: 
        x = np.mean(i[start:finish])
        activity.append(x)
    return activity

def inflection_point(traces, activity, activity_base, start, end):
    
    inflection_point_y = (np.array(activity) - np.array(activity_base))/2 + np.array(activity_base)
    
    inf = []
    for i in range(len(traces)):
        
        #calculate the difference array
        difference_array = np.absolute(traces[i][start:end]-inflection_point_y[i])
    
        # find the index of minimum element from the array
        index = difference_array.argmin() + start
        inf.append(index)
    return inf



filelistdg = glob.glob(r'file path.csv')
filelistmossy = glob.glob(r'file path.csv')

filenamedg = filelistdg[0]
filenamemossy = filelistmossy[0]

# windows unfiltered traces:
# M1 activity [500:1000] [1700:2518] inflection [1467:1819] minpeak [1360:1560] max peak [1640:1919]
# M2 activity [500:1000] [2205:2518] inflectiondg [2145:2219] inflectionmossy [2145:2219] no min or max peak
# M3 activity [500:1000] [2610:2989] inflectiondg [2491:2617] inflectionmossy [2491:2617] min peak [2411:2584], max peak [2557:2690]
# M4 activity [500:1000] [2698:2990] inflectiondg [2558:2711] inflectionmossy [2558:2711] min peak [2345:2684], max peak [2578:2817]
# M5 activity [500:1000] [2485:2997] inflectiondg [2352:2511] inflectionmossy [2352:2511] min peak [2338:2491], max peak [2378:2817]
    
#window size to calculate activity
startdg = 800
enddg = 1000

startmossy = 1000
endmossy = 1200

inf_startdg = 0
inf_enddg = 100

inf_startmossy = 0
inf_endmossy = 100

max_peak_start = 0
max_peak_end = 100

min_peak_start = 0
min_peak_end = 100

dfdg = pd.read_csv(filenamedg)
g = list(dfdg)
tracesdg = []
for i in g[1:]:
    g = list(dfdg[i])
    tracesdg.append(g)
np.save(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'tracesdg {}'.format(filenamemossy.split('\\')[-1].split('.')[0])), tracesdg)    

    
dfmossy = pd.read_csv(filenamemossy)
g = list(dfmossy)
tracesmossy = []
for i in g[1:]:
    g = list(dfmossy[i])
    tracesmossy.append(g)

np.save(os.path.join('\\'.join(filenamemossy.split('\\')[:-1]),'tracesmossy {}'.format(filenamemossy.split('\\')[-1].split('.')[0])), tracesmossy)    


plt.figure()
plt.plot(tracesdg[0])

plt.figure()
plt.plot(tracesmossy[0])

# Activity

activity_mossy_base = activity(tracesmossy, 800, 1000)
np.save(os.path.join('\\'.join(filenamemossy.split('\\')[:-1]),'activity_mossy_base {}'.format(filenamemossy.split('\\')[-1].split('.')[0])), activity_mossy_base)    

activity_mossy = activity(tracesmossy, startmossy, endmossy)
np.save(os.path.join('\\'.join(filenamemossy.split('\\')[:-1]),'activity_mossy {}'.format(filenamemossy.split('\\')[-1].split('.')[0])), activity_mossy)

activity_dg_base = activity(tracesdg, 600, 800)
np.save(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'activity_dg_base {}'.format(filenamedg.split('\\')[-1].split('.')[0])), activity_dg_base)    

activity_dg = activity(tracesdg, startdg, enddg)
np.save(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'activity_dg {}'.format(filenamedg.split('\\')[-1].split('.')[0])), activity_dg)

# determine activity deltaf/f avtivity seizure - activity base/activity base

deltaf_dg_activity =  (np.array(activity_dg) - np.array(activity_dg_base))/np.array(activity_dg_base)
deltaf_mossy_activity =  (np.array(activity_mossy) - np.array(activity_mossy_base))/np.array(activity_mossy_base)


# Inflection point

inflection_point_mossy = inflection_point(tracesmossy, activity_mossy, activity_mossy_base, inf_startmossy, inf_endmossy)
inflection_point_dg = inflection_point(tracesdg, activity_dg, activity_dg_base, inf_startdg, inf_enddg)

inflection_diff = np.mean(inflection_point_dg) - np.mean(inflection_point_mossy)
inflection_diff_sec = inflection_diff/15.253


# Onset

for i in range(len(tracesmossy)):
    plt.figure()
    plt.plot(tracesmossy[i])
    plt.plot(inflection_point_mossy[i],tracesmossy[i][inflection_point_mossy[i]], marker="o", markersize=10, markerfacecolor="red")


plt.plot(tracesmossy[0])


#Mean trace analysis using raw traces
adgmean = np.array(tracesdg)
resdgmean = np.average(adgmean, axis=0)
max_peakdg = list(resdgmean).index(np.max(resdgmean[max_peak_start:max_peak_end]))
min_peakdg = list(resdgmean).index(np.min(resdgmean[min_peak_start:min_peak_end]))

np.save(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'mean_trace_dg {}'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])), resdgmean)    

amossymean = np.array(tracesmossy)
resmossymean = np.average(amossymean, axis=0)
max_peakmossy = list(resmossymean).index(np.max(resmossymean[max_peak_start:max_peak_end]))
min_peakmossy = list(resmossymean).index(np.min(resmossymean[min_peak_start:min_peak_end]))

np.save(os.path.join('\\'.join(filenamemossy.split('\\')[:-1]),'mean_trace_mossy {}'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])), resmossymean)    

max_peak_diff = np.mean(max_peakdg) - np.mean(max_peakmossy)
max_peak_diff_sec = max_peak_diff/15.253

min_peak_diff = np.mean(min_peakdg) - np.mean(min_peakmossy)
min_peak_diff_sec = min_peak_diff/15.253

#std by column for all traces for dg and mossy
adgstd = np.array(tracesdg)
resdgstd = np.std(adgstd, axis=0)

np.save(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'std_trace_dg {}'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])), resdgstd)    

amossy = np.array(tracesmossy)
resmossystd = np.std(amossy, axis=0)

np.save(os.path.join('\\'.join(filenamemossy.split('\\')[:-1]),'std_trace_mossy {}'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])), resmossystd)

x = list(range(len(resdgmean)))
plt.figure()
plt.plot(x,resdgmean, label='DG')
plt.fill_between(x, resdgmean-resdgstd, resdgmean+resdgstd, alpha=0.2)

plt.plot(x,resmossymean, label='Mossy')
plt.fill_between(x, resmossymean-resmossystd, resmossymean+resmossystd, alpha=0.2)
plt.legend()
plt.savefig(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'mean traces figure {}.svg'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])))


print('activity dg seizure: ' + str(deltaf_dg_activity))
print('activity mossy seizure: ' + str(deltaf_mossy_activity))
print('inflection diff: ' + str(inflection_diff_sec))
print('max peak diff: ' + str(max_peak_diff_sec))
print('min peak diff: ' + str(min_peak_diff_sec))

# plt.figure()
# plt.plot(x,resmossymean, label='Mossy')
# plt.plot(np.mean(inflection_point_mossy), resdgmean[int(np.mean(inflection_point_mossy))], marker="o", markersize=8, markerfacecolor="red", label='inflection point mossy')  
# plt.plot(x,resdgmean, label='DG')
# plt.plot(np.mean(inflection_point_dg), resdgmean[int(np.mean(inflection_point_dg))], marker="o", markersize=8, markerfacecolor="blue", label='inflection point mossy')  

# plt.legend()
# plt.savefig(os.path.join('\\'.join(filenamedg.split('\\')[:-1]),'mean traces figure {}.svg'.format(filenamemossy.split('\\')[-1].split('.')[0][:-5])))