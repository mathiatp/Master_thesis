import numpy as np


time_flatten_ravel = np.load('Profiler_stats/time_flatten_ravel.npy')*1000

avg_flatten = np.average(time_flatten_ravel[:,0],0)
avg_ravel = np.average(time_flatten_ravel[:,1],0)



min_flatten = np.min(time_flatten_ravel[:,0],0)
min_ravel = np.min(time_flatten_ravel[:,1],0)



max_flatten = np.max(time_flatten_ravel[:,0],0)
max_ravel = np.max(time_flatten_ravel[:,1],0)



percetile_flatten = np.percentile(time_flatten_ravel[:,0],95,0)
percetile_ravel = np.percentile(time_flatten_ravel[:,1],95,0)

print('finito')