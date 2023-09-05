
import torch
import torch.nn as nn
import os
import time
from dataclasses import dataclass
import yaml

NN = 'NN'
NT = 'NT'
TN = 'TN'
TT = 'TT'

gpu_peak = {
    ('NVIDIA A100-PCIE-40GB', torch.float16): 312e12,
    ('Tesla V100-PCIE-32GB', torch.float16): 125e12,
    ('NVIDIA A100-PCIE-40GB', torch.float16): 312e12,

}

current_gpu = torch.cuda.get_device_name(0)
current_gpu_peak = gpu_peak[(current_gpu, torch.float16)]

print(f'GPU: {current_gpu}')
print(f'    Peak Compute: {current_gpu_peak / 1e12:.3f} TFLOPS')
print()


batch_ranges = {
    'bert-large': [1, 32],
    'gpt3':       [1, 1],
    'mgn-cloth':  [1, 1],
    'muzero':     [1, 1024],
    'qdtrack':    [1, 2],
    'swinv2-b-p': [1, 32],
    'tabnet':     [2, 128],
    'tacotron2':  [1, 64],
    'wavenet':    [1, 1],
    'pinn-ns':    [1, 1],
    'nerf':       [1024, 1024],
}

def benchmark(f, *args, flops=1, NI=None):
    torch.backends.cudnn.benchmark = False

    t0 = time.perf_counter()
    f(*args)
    t1 = time.perf_counter()
    t_est = t1 - t0

    if NI is None:  NI = int(30 / t_est)

    NI = int(os.environ.get('NITERS', NI))
    assert NI is not None and NI > 0, f'NI={NI}, flops={flops}'
    print(f'    Num Iters: {NI}')

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    ev_start.record()
    for _ in range(NI): f(*args)
    ev_end.record()
    torch.cuda.synchronize()

    tt = (ev_start.elapsed_time(ev_end) / 1000)

    print(f'    Avg Latency: {1000 * tt / NI:.3f} ms, {flops * NI / tt / 1e9:.3f} GFLOPS')
    print(f'    Compute: {flops * NI / tt / current_gpu_peak * 100:.2f} % of peak')

    return flops * NI / tt


@dataclass(frozen=True)
class Matmul:
    M : int
    N : int
    K : int
    layout : str

    @property
    def tr_a(self): return self.layout[0] == 'T'

    @property
    def tr_b(self): return self.layout[1] == 'T'

    def benchmark(self, dev, dtype):
        if not self.tr_a: A = torch.randn(self.M, self.K, device=dev, dtype=dtype)
        else: A = torch.randn(self.K, self.M, device=dev, dtype=dtype).t()

        if not self.tr_b: B = torch.randn(self.K, self.N, device=dev, dtype=dtype)
        else: B = torch.randn(self.N, self.K, device=dev, dtype=dtype).t()

        return benchmark(torch.matmul, A, B, flops=2 * self.M * self.N * self.K)

def Linear(M, N, K): return Matmul(M, N, K, NT)

@dataclass(frozen=True)
class BatchMatmul:
    L : int
    M : int
    N : int
    K : int
    layout : str

    @property
    def tr_a(self): return self.layout[0] == 'T'

    @property
    def tr_b(self): return self.layout[1] == 'T'

    def benchmark(self, dev, dtype):
        if not self.tr_a: A = torch.randn(self.L, self.M, self.K, device=dev, dtype=dtype)
        else: A = torch.randn(self.L, self.K, self.M, device=dev, dtype=dtype).transpose(-1, -2)

        if not self.tr_b: B = torch.randn(self.L, self.K, self.N, device=dev, dtype=dtype)
        else: B = torch.randn(self.L, self.N, self.K, device=dev, dtype=dtype).transpose(-1, -2)

        return benchmark(torch.bmm, A, B, flops=2 * self.L * self.M * self.N * self.K)

@dataclass(frozen=True)
class Conv2D:
    B : int
    C : int
    K : int
    H : int
    W : int
    P : int
    Q : int
    R : int
    S : int
    stride : int

    def benchmark(self, dev, dtype):
        conv = torch.nn.Conv2d(self.C, self.K, (self.R, self.S), stride=self.stride, padding=0).to(dev).to(dtype)
        x = torch.randn(self.B, self.C, self.H, self.W, device=dev, dtype=dtype)
        return benchmark(conv, x, flops=2 * self.B * self.C * self.K * self.P * self.Q * self.R * self.S)

