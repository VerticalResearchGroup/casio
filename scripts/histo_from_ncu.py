import pandas as pd
import sys
import tempfile
import utils




headers=['cumm-fraction-time', 'metric', 'fraction-time'] + utils.launch_stats

def get_histograms(ncu_raw_file):
    df = pd.read_csv(
        utils.Reader(utils.read_ncu_file(ncu_raw_file)),
        low_memory=False,
        thousands=r',')

    print(df)

    df2 = df.filter(utils.stats_of_interest, axis=1)
    df3 = df.filter(utils.launch_stats, axis=1)

    averages = {}
    flamegraphs = {}
    nbins = 50
    total_cycles = 0

    for i,y in enumerate(df2['gpc__cycles_elapsed.max']):
        if (i != 0):
            # y = y.replace(",", "")
            total_cycles = total_cycles + int(y)

    for x in utils.stats_of_interest:
        if (x == 'gpc__cycles_elapsed.max'):
            continue
        avg = 0.0
        flamegraphs[x] = []

        running_c = 0
        for i, y in enumerate(df2[x]):
            if i == 0: continue
            bin = int(float(y)/(100/nbins))
            if bin > nbins:
                print("error ", y)
            else:
                c = int(df2['gpc__cycles_elapsed.max'][i])
                f = c/total_cycles
                avg = avg + f * float(y)

                running_c += c
                g = running_c/total_cycles
                z = [g,float(y),f]
                for l in utils.launch_stats:
                    z.append(str(df3[l][i]))
                flamegraphs[x].append(','.join(map(lambda x: f'"{x}"', z)))
        averages[x] = avg

    return averages, flamegraphs

assert len(sys.argv) == 3, 'Usage: histo_from_ncu.py ncu_raw_file.txt outputprefix'

averages, flamegraphs = get_histograms(sys.argv[1])
output_prefix = sys.argv[2]

with open(f'{output_prefix}feature-avg.csv', 'w') as avg_file:
    for x in utils.stats_of_interest:
        if x in utils.ignore_list: continue

        print(x)

        with open(f'{output_prefix}flame.{x}.csv', 'w') as f:
            print(','.join(headers), file=f)
            for j in flamegraphs[x]: print(j, file=f)
            f.close()

        print(f'{x}, {averages[x]}', file=avg_file)
