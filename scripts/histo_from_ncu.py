import pandas as pd
import sys
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


stats_of_interest = ['gpc__cycles_elapsed.max', 'sm__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'l1tex__throughput.avg.pct_of_peak_sustained_active', 'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'sm__issue_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed', 'sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']

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
    histograms = {}
    averages = {}
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
        histograms[x] = [0 for i in range(nbins+1)] 
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
        averages[x] = avg
#        print(x, histograms[x])
    return histograms, averages

histograms1, averages1 = get_histograms(sys.argv[1])
histograms2, averages2 = get_histograms(sys.argv[2])

for x in stats_of_interest:
    if (x == 'gpc__cycles_elapsed.max'):
        continue
    print("feature ", x, averages1[x], averages2[x])
    for i,y in enumerate(histograms1[x]):
        if (y == 0):
            rel_err = 0
        else:
            rel_err = (y - histograms2[x][i])/y
#        print(i, y, histograms2[x][i], rel_err)


