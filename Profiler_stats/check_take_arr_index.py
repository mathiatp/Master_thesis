# Check the difference between np.take() and normal array indexing
# test with 500 frames

import numpy as np

time_arr = np.load('Profiler_stats/time_take_arr_index.npy')

avg_take = np.average(time_arr[:,0],0)
avg_arr_index = np.average(time_arr[:,1],0)

min_take = np.min(time_arr[:,0],0)
min_arr_index = np.min(time_arr[:,1],0)

max_take = np.max(time_arr[:,0],0)
max_arr_index = np.max(time_arr[:,1],0)

percetile_take = np.percentile(time_arr[:,0],95,0)
percetile_arr_index = np.percentile(time_arr[:,1],95,0)
pass