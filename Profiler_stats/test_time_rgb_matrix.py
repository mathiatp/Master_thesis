import numpy as np


time_np_all_short_long = np.load('Profiler_stats/time_all_short_long.npy')*1000

avg_np_all = np.average(time_np_all_short_long[:,0],0)
avg_short = np.average(time_np_all_short_long[:,1],0)
avg_long = np.average(time_np_all_short_long[:,2],0)


min_np_all = np.min(time_np_all_short_long[:,0],0)
min_short = np.min(time_np_all_short_long[:,1],0)
min_long = np.min(time_np_all_short_long[:,2],0)


max_np_all = np.max(time_np_all_short_long[:,0],0)
max_short = np.max(time_np_all_short_long[:,1],0)
max_long = np.max(time_np_all_short_long[:,2],0)


percetile_np_all = np.percentile(time_np_all_short_long[:,0],95,0)
percetile_short = np.percentile(time_np_all_short_long[:,1],95,0)
percetile_long = np.percentile(time_np_all_short_long[:,2],95,0)
print('finito')