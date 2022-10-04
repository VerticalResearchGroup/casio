
import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile
from torch_wrapper import benchmark_wrapper
import params
import torch

NN = 2000
NE = 10000

latent_size = 128
output_size = 128
num_layers = 2

##################################################
class Mlp(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            torch.nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

        self.norm = torch.nn.LayerNorm(dims[-1])

    def forward(self, x):
        return self.norm(self.layers(x))

class GraphNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp = Mlp([latent_size] * num_layers + [output_size])
        self.node_mlp = Mlp([latent_size] * num_layers + [output_size])

    def forward(self, x):
        return (self.node_mlp(x[0]), self.edge_mlp(x[1]))

class MeshGraphNet(torch.nn.Module):
    def __init__(self, nblk):
        super().__init__()
        self.blocks = torch.nn.Sequential(*[
            GraphNetBlock()
            for _ in range(nblk)
        ])

    def forward(self, v, e):
        return self.blocks((v, e))
##################################################

net = MeshGraphNet(15)

v = torch.randn(NN, latent_size)
e = torch.randn(NE, latent_size)

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

def roi():
    opt.zero_grad()
    out = net(v, e)
    loss = out[0].sum() + out[1].sum()
    loss.backward()
    opt.step()

benchmark_wrapper('meshgn', roi)
