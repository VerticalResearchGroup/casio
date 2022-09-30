import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
import time
import sys
import cudaprofile

torch.backends.cudnn.benchmark = False

B = 408
N = 50
M = 20
print(f'Batch Size: {B}, Iters: {N}')
dev = torch.device('cuda:1')

net = torchvision.models.resnet50(pretrained=False).half()
net.eval().to(dev)

# opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
x = torch.randn((B, 3, 224, 224), device=dev, dtype=torch.float16)

print(f'Warmup with {N} Iters')
torch.cuda.synchronize()
for i in range(N):
    # opt.zero_grad()
    # x = torch.randn((B, 3, 224, 224), device=dev, dtype=torch.float16)
    net(x)
    # loss = torch.sum(yp)
    # loss.backward()
    # opt.step()
torch.cuda.synchronize()

print(f'Running {M} Iters')


t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    net(x) # Region of Interest

    torch.cuda.synchronize()
t1 = time.perf_counter()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=100))

print(f'Total Time: {t1 - t0}')
print(f'Throughput: {M*B / (t1 - t0)}')
