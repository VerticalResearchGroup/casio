import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from . import rnnt
from . import librespeech

device = torch.device(params.devname)
net = rnnt.Rnnt()
net.to(params.dtype_torch).to(device)

dataset = librespeech.Librespeech(f'{params.casio}/rnnt/librespeech-min/librespeech-min.json')

x = torch.Tensor(dataset[0].audio.samples)[:int(239*186560/389)].unsqueeze_(0)
l = torch.LongTensor([int(239*186560/389)])

net(x, l)

opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def roi():
    opt.zero_grad()
    yp = net(x, l)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()

benchmark_wrapper2(roi)
