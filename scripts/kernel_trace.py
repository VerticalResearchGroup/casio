import pandas as pd
import sys
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


stats_of_interest = ['Kernel Name', 'gpc__cycles_elapsed.max', 'launch__grid_size']
#, 'sm__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'l1tex__throughput.avg.pct_of_peak_sustained_active', 'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'sm__issue_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed', 'sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed']
ignore_list = [stats_of_interest[0]]

def get_kerneltrace(filename):
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
    df2 = df.filter(stats_of_interest, axis=1)
    for i,y in enumerate(df2['gpc__cycles_elapsed.max']):
        if (i == 0):
            continue
        if (i != 0):
            y = y.replace(",", "")
        print(df2['Kernel Name'][i], y, df2['launch__grid_size'][i])
get_kerneltrace(sys.argv[1])

