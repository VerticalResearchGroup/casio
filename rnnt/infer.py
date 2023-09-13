import torch
import torchvision

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from . import rnnt
from . import librespeech

device = torch.device(params.devname)
net = rnnt.Rnnt()
net.to(params.dtype_torch).to(device).eval()

dataset = librespeech.Librespeech(f'{params.casio}/rnnt/librespeech-min/librespeech-min.json')

x = torch.Tensor(dataset[0].audio.samples)[:int(239*186560/389)] \
    .to(params.dtype_torch).to(device).unsqueeze_(0)

l = torch.LongTensor([int(239*186560/389)]).to(device)

def roi():
    _ = net(x, l)

benchmark_wrapper2(roi)
