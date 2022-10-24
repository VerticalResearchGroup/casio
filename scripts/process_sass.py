import pandas as pd
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
filename=sys.argv[1]
outprefix=sys.argv[2]
outfile = ""
list_of_kernels = {}
with open(filename) as fp:
   line = fp.readline()
   cnt = 1
   while line:
       if line.startswith('"Kernel Name"'):
           if (cnt != 1):
              f.close()
           outfilename = outprefix + str(cnt) + "-kernelname.txt"

           print('\rCreating %s' % outfilename, end=" ")
           sys.stdout.flush()
           time.sleep(0.01)
           f0 = open(outfilename, 'w')
           f0.write(line)
           f0.close()
           outfilename = outprefix + str(cnt) + "-sass.csv"
           f = open(outfilename, 'w')
           kname = line.replace('"Kernel Name",','').strip()
           if kname not in list_of_kernels:
               list_of_kernels[kname] = 1
           else:
               list_of_kernels[kname] += 1
           cnt += 1
       else:
           try: 
               f
               if not f.closed:
                   f.write(line)
           except NameError: 
               line = fp.readline()
               continue
       line = fp.readline()
   f.close()
   print(" ")

addr_map = {}
opcode_map = {}
total_inst = 0
for i in range(cnt-1):
    filename = outprefix + str(i+1) + "-sass.csv"
    print('\rReading %s' % filename, end=" ")
    sys.stdout.flush()
    time.sleep(0.01)
    df = pd.read_csv(filename, low_memory=False, thousands=r',')
    df2 = df.filter(["Address","Source","Thread Instructions Executed","Predicated-On Thread Instructions Executed"], axis=1)
    # print(df2)
    # static count contribution
    for j,x in enumerate(df2['Address']):
        if x in addr_map.keys():
            addr_map[x] += df2["Thread Instructions Executed"][j]
        else:
            addr_map[x] = df2["Thread Instructions Executed"][j]
        total_inst += df2["Thread Instructions Executed"][j]
        inst_assembly = df2["Source"][j]
        inst_assembly = inst_assembly[5:]
        opcode = inst_assembly.split(' ')[0].strip()
        if (len(opcode) <= 1):
            inst_assembly = df2["Source"][j]
            inst_assembly = inst_assembly[6:]
            opcode = inst_assembly.split(' ')[0].strip()
        if (len(opcode) <= 1):
            print("Something went wrong....could not get opcode ", df2["Source"][j])
            quit()
        if df2["Thread Instructions Executed"][j] == 0:
            continue
        if opcode in opcode_map:
            opcode_map[opcode] += df2["Thread Instructions Executed"][j]
        else:
            opcode_map[opcode] = df2["Thread Instructions Executed"][j]
#    print(filename)
print(" ")
for x in addr_map.keys():
    addr_map[x] = addr_map[x]/total_inst
for x in opcode_map.keys():
    opcode_map[x] = opcode_map[x]/total_inst
f = open(outprefix + "opcode-dist.csv", 'w')
sum = 0
for key, value in dict(sorted(opcode_map.items(), key=lambda item: item[1], reverse=True )).items():
    sum = sum + value
    print(key, ',',"{:2.10f}".format(value),',',"{:2.10f}".format(sum),  file=f)
f.close()
f = open(outprefix + "kernelname-dist.csv", 'w')
for key, value in dict(sorted(list_of_kernels.items(), key=lambda item: item[1], reverse=True )).items():
    print(key, '|', value, file=f)
f.close()
f = open(outprefix + "inst-dist.csv", 'w')
sum = 0
n = len(addr_map.keys())
i = 1
for key, value in dict(sorted(addr_map.items(), key=lambda item: item[1], reverse=True )).items():
    sum = sum + value
    icount = value * total_inst
    print(key, ',',"{:2.10f}".format(i/n),',',"{:2.10f}".format(value),',',"{:2.10f}".format(sum),',',icount, file=f)
    i = i + 1
f.close()
f = open(outprefix + "inst-total.csv", 'w')
print("Total inst: ", total_inst, file=f)
f.close()
