import torch
import torch.fx
import ssdrn34
import ssdrn34_weights
import perftools

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

from torch.fx.node import Node, map_aggregate


# ssd = ssdrn34.SsdRn34()
# ssd.load_weights('ssdrn34.npz')
ssd = ssdrn34.SsdRn34_300()

gm = torch.fx.symbolic_trace(ssd, dict(x=torch.zeros(1, 3, 300, 300)))
x = torch.rand(1, 3, 300, 300)
opt = torch.optim.SGD(ssd.parameters(), lr=0.001, momentum=0.9)


opt.zero_grad()
yp = ssd(x)
loss = yp[0].sum() + yp[1].sum()
loss.backward()
opt.step()


