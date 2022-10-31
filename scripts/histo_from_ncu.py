import pandas as pd
import sys
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


stats_of_interest = ['gpc__cycles_elapsed.max', 'sm__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'l1tex__throughput.avg.pct_of_peak_sustained_active', 'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'sm__issue_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed', 'sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']
ignore_list = [stats_of_interest[0]]
launch_stats = ['Kernel Name', 'launch__block_dim_x','launch__block_dim_y','launch__block_dim_z','launch__block_size', 'launch__grid_dim_x','launch__grid_dim_y','launch__grid_dim_z','launch__grid_size','launch__occupancy_limit_blocks','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','launch__occupancy_limit_warps','launch__occupancy_per_block_size','launch__occupancy_per_register_count','launch__occupancy_per_shared_mem_size','launch__registers_per_thread','launch__registers_per_thread_allocated','launch__shared_mem_config_size','launch__shared_mem_per_block','launch__shared_mem_per_block_allocated','launch__shared_mem_per_block_driver','launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_static','launch__thread_count','launch__waves_per_multiprocessor']

def get_histograms(filename):
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
    df3 = df.filter(launch_stats, axis=1)
    histograms = {}
    averages = {}
    flamegraphs = {}
    nbins = 50
    total_cycles = 0
    for i,y in enumerate(df2['gpc__cycles_elapsed.max']):
        if (i != 0):
            y = y.replace(",", "")
            total_cycles = total_cycles + int(y)
    for x in stats_of_interest:
        if (x == 'gpc__cycles_elapsed.max'):
            continue
        avg = 0.0
        flamegraphs[x] = []
        histograms[x] = [0 for i in range(nbins+1)] 
        running_c = 0
        for i,y in enumerate(df2[x]):
            if (i == 0):
                # throw away 1st row
                continue
            bin = int(float(y)/(100/nbins))
            if bin > nbins:
                print("error ", y)
            else:
                c = int(df2['gpc__cycles_elapsed.max'][i].replace(",",""))
                f = c/total_cycles
                avg = avg + f * float(y)
                histograms[x][bin] = histograms[x][bin]+f
                running_c += c
                g = running_c/total_cycles
                s = ''
                for l in launch_stats:
                       s = s + str(df3[l][i])+' '	
                z = [g,float(y),f,s]
                flamegraphs[x].append(z)
        averages[x] = avg
#        print(x, histograms[x])
    return histograms, averages, flamegraphs
if len(sys.argv) != 4:
    print("need 3 args: file1 file2 outputprefix")
histograms1, averages1, flamegraphs1 = get_histograms(sys.argv[1])
histograms2, averages2, flamegraphs2 = get_histograms(sys.argv[2])
output_prefix = sys.argv[3]
filename = output_prefix + 'feature-avg' + '.csv'
f0 = open(filename, 'w')

for x in stats_of_interest:
#    if (x == 'gpc__cycles_elapsed.max'):
    if (x in ignore_list):
        continue
    if averages1[x] == 0:
        continue
    print(x)
    filename = output_prefix + 'flame.' + x + '.csv'
    f = open(filename, 'w')
    for j in flamegraphs1[x]:
        print(j[0], j[1],j[2],j[3],file=f)
    f.close()
    rel_err = 100*(averages1[x] - averages2[x])/averages1[x]
    print(x, averages1[x], averages2[x], "{:2.10f}".format(rel_err), file=f0)
    for i,y in enumerate(histograms1[x]):
        if (y == 0):
            rel_err = 0
        else:
            rel_err = (y - histograms2[x][i])/y
#        print(i, y, histograms2[x][i], rel_err)

f0.close()
