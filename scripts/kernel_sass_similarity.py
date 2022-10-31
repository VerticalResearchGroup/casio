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
    fname=prefix+"_"+str(i)+"-sass.csv"
    k_filename=prefix+"_"+str(i)+"-kernelname.txt"
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
quit()
drop_list = []
pdb.set_trace()
k=np.arange(0, len(df2), 1).tolist()
df2 = pdist(df2, 'cosine')
df4 = pd.DataFrame(squareform(df2), columns=k, index=k)
print(df4)
plt.figure(figsize=(10,10))
sns.heatmap(df4)
plt.show()
quit()
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
