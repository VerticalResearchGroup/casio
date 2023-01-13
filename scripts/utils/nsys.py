from .common import *
from .is_gemm import *
from . import cache

@dataclass
class NsysKernel:
    name : str
    time_ns : float
    num_threads : int

    @property
    def is_gemm(self): return is_gemm(self.name)

    def __repr__(self):
        return f'Kernel(name={shorten_string(self.name)}, {self.num_threads} threads, {self.time_ns}ns )'

nsys_trace_regex = r'(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*),(\d*\.?\d*),(\d*\.?\d*),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\"?([^"]+)\"?'

def get_nsys_gputrace_file(plat : str, app : str, batch : int):
    return f'{CASIO}/casio-results/summaries/{plat}/{app}/batch-{batch}_gputrace.csv.gz'

def parse_nsys_line(line):
    m = re.match(nsys_trace_regex, line.strip())
    if m is None:
        assert False, f'Failed to parse line: {line}'

    for i in [4, 5, 6, 7, 8, 9]:
        if m.group(i) == '': return None

    num_threads = np.prod([int(m.group(i)) for i in [4, 5, 6, 7, 8, 9]])
    kname = m.group(13)
    return NsysKernel(kname.strip(), float(m.group(2)), num_threads)

def read_nsys_trace(nsys_trace_file):
    with gzip.open(nsys_trace_file,'rt') as f:
        next(f)
        return list(
            filter(
                lambda x: x is not None,
                map(
                    parse_nsys_line,
                    filter(
                        lambda line: not is_blacklisted(line),
                        f))))


def parse_nsys_kernsum(line):
    # Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    regex = r'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),(.*)'
    m = re.match(regex, line)
    assert m is not None, f'Failed to parse line: "{line}"'
    return m.group(9)

# Mike is in a hurry here. I'm sorry for the duplicate code
def parse_nsys_kernsum2(line):
    # Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    regex = r'([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),(.*)'
    m = re.match(regex, line)
    assert m is not None, f'Failed to parse line: "{line}"'
    return float(m.group(2)), int(m.group(3)), m.group(9)

def get_nsys_niter(plat, app, batch):
    nsys_file = None
    for filename in glob.glob(f'{CASIO}/casio-results/{plat}/{app}/nsys*b{batch}-*.nsys-rep'):
        nsys_file = filename
        break

    assert nsys_file is not None, f'Failed to find nsys file for {plat}/{app} batch {batch}'

    return int(nsys_file.replace('.nsys-rep', '').split('-')[-1][1:])


def nsys_get_kernels(app, plat):
    batch = get_large_batch_size(plat, app)
    nsys_trace_file = get_nsys_gputrace_file(plat, app, batch)
    return read_nsys_trace(nsys_trace_file)

@cache.cache_list(float)
def nsys_get_gemm_times_internal(plat):
    times = []
    for app in apps:
        prettyname = app_pretty_names[app]
        print(f'Processing {prettyname}...')

        kernels = nsys_get_kernels(app, plat)
        tot_time = sum(k.time_ns for k in kernels)
        gemm_time = sum(k.time_ns for k in kernels if k.is_gemm)
        times.append(gemm_time)
        times.append(tot_time)

    return times

def nsys_get_gemm_times(plat):
    times = nsys_get_gemm_times_internal(plat)
    return list(zip(times[::2], times[1::2]))
