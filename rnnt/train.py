import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from . import rnnt
from . import librespeech

device = torch.device(params.devname)
net = rnnt.Rnnt()
net.to(params.dtype_torch).to(device).train()

dataset = librespeech.Librespeech(f'{params.casio}/rnnt/librespeech-min/librespeech-min.json')


x = torch.Tensor(dataset[0].audio.samples)[:int(239*186560/389)] \
    .to(params.dtype_torch).unsqueeze_(0).expand(params.bs, -1).to(device)

l = torch.LongTensor([int(239*186560/389)] * params.bs).to(device)

# print(dataset[0].transcript)
y = torch.LongTensor(list(map(
    lambda c: rnnt.Rnnt.labels.index(c),
    list(dataset[0].transcript)))) \
        .unsqueeze(0).expand(params.bs, -1).to(device)

# print(y)

net(x, l, y)

opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    yp = net(x, l, y)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()

benchmark_wrapper2(roi)
