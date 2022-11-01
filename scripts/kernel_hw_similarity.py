import pandas as pd
import pdb
import sys
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
stats_of_interest = ['sm__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'l1tex__throughput.avg.pct_of_peak_sustained_active', 'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'sm__issue_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed', 'sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']
ignore_list = [stats_of_interest[0]]
launch_stats = ['Kernel Name', 'launch__block_dim_x','launch__block_dim_y','launch__block_dim_z','launch__block_size', 'launch__grid_dim_x','launch__grid_dim_y','launch__grid_dim_z','launch__grid_size','launch__occupancy_limit_blocks','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','launch__occupancy_limit_warps','launch__occupancy_per_block_size','launch__occupancy_per_register_count','launch__occupancy_per_shared_mem_size','launch__registers_per_thread','launch__registers_per_thread_allocated','launch__shared_mem_config_size','launch__shared_mem_per_block','launch__shared_mem_per_block_allocated','launch__shared_mem_per_block_driver','launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_static','launch__thread_count','launch__waves_per_multiprocessor']

def get_df(filename):
    # create a temp file with just csv contents to pass to pandas.read_csv
    tmp = tempfile.NamedTemporaryFile()
    ft = open(tmp.name, 'w')
    with open(filename) as fp:
        line = fp.readline()
        data_started = False
        while line:
            if line.startswith('"ID","Process ID","Process Name",'):
                data_started = True
            if data_started:
                ft.write(line)
            line = fp.readline()
    ft.seek(0)
    ft = open(tmp.name, 'r')
    df = pd.read_csv(ft, low_memory=False, thousands=r',')
    ft.close()
    column_names = list(df.columns.values)
    #print(*column_names, sep='\n')
    #quit()
    # 'launch__thread_count', 'launch__block_size', 'launch__thread_count', 'launch__waves_per_multiprocessor', 'launch__registers_per_thread']
    df2 = df.filter(stats_of_interest, axis=1)
    df2 = df2.drop([0], axis=0)
    df3 = df.filter(['Kernel Name'], axis=1).drop([0], axis=0)
    return df2, df3
df, df_names = get_df(sys.argv[1])
if (len(sys.argv) == 3):
    print("reading second array")
    df2, df2_names = get_df(sys.argv[2])
    df = pd.concat([df, df2], ignore_index=True, sort=False)
    df_names = pd.concat([df_names, df2_names], ignore_index=True, sort=False)
pd.set_option('display.max_rows', 10)
frame_len = len(df)
k=np.arange(0, frame_len, 1).tolist()
df2=pdist(df, 'cosine')
df4=pd.DataFrame(squareform(df2), columns=k, index=k)
#plt.figure(figsize=(10,10))
#sns.heatmap(df4)
#plt.show()

keep_list = set()
ignore_list = set()
for i in range(0, frame_len):
   if (i % 100 == 0):
      print("\r", i, " of ", frame_len, 'found ', len(keep_list), end='')
   if (i in ignore_list):
      continue
   for j in range(0, frame_len):
      x = df4.loc[i,j]
      if (x > 0.05):
         if (j in ignore_list):
            continue
         else:
            keep_list.add(i)
      else:
         ignore_list.add(j)
print()
print(keep_list)
set_of_kernel_names = set()
df2 = df
for x in enumerate(df_names['Kernel Name']):
   set_of_kernel_names.add(x[1])
print("Total kernels: ", frame_len)
print("Total uniq kernels: ", len(set_of_kernel_names))
print("Representative kernels: ", len(keep_list))

df2 = df
for i in range(0, frame_len):
   if (i in keep_list):
      pass
   else:
      print('Dropping ', i)
      df2 = df2.drop([i], axis=0)
print(len(df2))
k=np.arange(0, len(df2), 1).tolist()
df2 = pdist(df2, 'cosine')
df4 = pd.DataFrame(squareform(df2), columns=k, index=k)
print(df4)
plt.figure(figsize=(10,10))
sns.heatmap(df4)
plt.show()
for x in keep_list:
   print('rep-kernel-id: ', x, df.loc[x+1,:].to_list(), df_names['Kernel Name'][x+1])
for x in set_of_kernel_names:
   print("Uniq-kernel-name: ", x)

quit()


# move to long form
long_form = df4.unstack()

# rename columns and turn into a dataframe
long_form.index.rename(['Kernel A', 'Kernel B'], inplace=True)
long_form = long_form.to_frame('cosine distance').reset_index()

ignore_list = set() 
keep_list = set()
for i,x in enumerate(long_form['Kernel A']):
   if (i % 100 == 0):
       print("\r", i, " of ", len(long_form['Kernel A']), end='')
   if x in ignore_list:
#      print(x, 'is to be ignored')
      continue
   if not x in keep_list:
#      print('found ', x)
      keep_list.add(x)
   c = long_form['cosine distance'][i]
   if (c < 0.05):
      if (long_form['Kernel B'][i] != long_form['Kernel A'][i]):
         ignore_list.add(long_form['Kernel B'][i])
#         print('Found similar kernel ', long_form['Kernel B'][i], 'to ', x, c)
#         print(ignore_list)
#      else:
#         print('keeping ', long_form['Kernel B'][i])

print()
print(keep_list)
print(len(keep_list), " ", frame_len)
#new_list = set()
#for i,a in enumerate(long_form['Kernel A']):
#   b = long_form['Kernel B'][i]
#   c = long_form['cosine distance'][i]
#   if (a == b):
#      continue
#   if a in keep_list:
#      if (c < 0.05):
#         new_list.add(b)
#print(len(new_list))
for x in keep_list:
   print(x, df.loc[x+1,:].to_list(), df_names['Kernel Name'][x+1])
