import pandas as pd
import sys
import time
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

filename = sys.argv[1]
outprefix = sys.argv[2]

kernels = utils.parse_ncu_sass(filename)
addr_map, opcode_map, total_dyn_inst = utils.ncu_sass_stats(kernels)

for x in addr_map.keys():
    addr_map[x] = addr_map[x] / total_dyn_inst

for x in opcode_map.keys():
    opcode_map[x] = opcode_map[x] / total_dyn_inst


f = open(outprefix + "opcode-dist.csv", 'w')
csum = 0
for opcode, frac in dict(sorted(opcode_map.items(), key=lambda item: item[1], reverse=True)).items():
    csum = csum + frac
    print(f'{opcode}, {frac}, {csum}', file=f)
f.close()


# f = open(outprefix + "kernelname-dist.csv", 'w')
# for key, value in dict(sorted(list_of_kernels.items(), key=lambda item: item[1], reverse=True )).items():
#     print(key, '|', value, file=f)
# f.close()

f = open(outprefix + "inst-dist.csv", 'w')
csum = 0
num_pcs = len(addr_map)
for i, (pc, frac) in enumerate(dict(sorted(addr_map.items(), key=lambda item: item[1], reverse=True)).items()):
    csum = csum + frac
    icount = frac * total_dyn_inst
    print(f'{pc}, {(i + 1) / num_pcs}, {frac}, {csum}, {icount}', file=f)
f.close()


f = open(outprefix + "inst-total.csv", 'w')
print("Total inst: ", total_dyn_inst, file=f)
f.close()
