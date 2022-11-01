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
from os.path import exists


stats_of_interest = ['Source', 'Instructions Executed']

def get_df(filename):
    df = pd.read_csv(filename, low_memory=False, thousands=r',')
    df2 = df.filter(stats_of_interest, axis=1)
    c = 0
    for i, x in enumerate(df2['Instructions Executed']):
        c = c + x
    p_time=[]
    for i, x in enumerate(df2['Instructions Executed']):
        p_time.append(x*1.0/c)
    df2['p-time'] = p_time
    opcode_list = []
    for i, x in enumerate(df2['Source']):
       inst_assembly = x
       inst_assembly = inst_assembly[5:]
       opcode = inst_assembly.split(' ')[0].strip()
       if (len(opcode) <= 1):
           inst_assembly = x
           inst_assembly = inst_assembly[6:]
       opcode = inst_assembly.split(' ')[0].strip()
       if (len(opcode) <= 1):
           print("Something went wrong....could not get opcode ", df2["Source"][j])
           quit()
       opcode_list.append(opcode)
    df2['opcode']=opcode_list
    rows_to_drop = set()
    for i, x in enumerate(df2['opcode']):
       for j, y in enumerate(df2['opcode']):
           if j in rows_to_drop:
              continue
           if (j <= i):
              pass
           else:
              if (x==y):
                 df2['p-time'][i] += df2['p-time'][j]
                 rows_to_drop.add(j)
    df2=df2.drop(list(rows_to_drop)).reset_index(drop=True)
    df3=df2.filter(['opcode', 'p-time'], axis=1)
    return(df3)
prefix=sys.argv[1]
nfiles=int(sys.argv[2])
kernel_names_list = []
for i in range(1,nfiles+1):
    fname=prefix+str(i)+"-sass.csv"
    k_filename=prefix+str(i)+"-kernelname.txt"
    if exists(fname):
       pass
    else:
       break
    with open(k_filename, 'r') as file:
       data = file.read().replace('\n', '')
    data = data.split('"')[3]
    kernel_names_list.append(data)
    print(fname)
    df=get_df(fname)
    if i == 1:
       df_full = df
    else:
       df_full = pd.merge(df, df_full, how='outer', on='opcode')
df_full = df_full.replace(float("NaN"), 0)
opcode_list = df_full['opcode']
df = df_full.drop(['opcode'], axis=1).T
# change name back to df
print(df)
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
print("Total kernels: ", frame_len)
print("Representative kernels: ", len(keep_list))
for a in keep_list:
   print('rep-kernel: ', a, kernel_names_list[a], end='' )
   l=[]
   for i,x in enumerate( df.iloc[a].tolist()):
      if (x >0.0):
         l.append(opcode_list[i] + ":" + str(round(x, 3)))
   print(' ', l) 

