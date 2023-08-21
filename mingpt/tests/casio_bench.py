import os
import sys

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import params
import cudaprofile
from torch_wrapper import benchmark_wrapper

from mingpt.model import GPT

from mingpt.model import GPT3SingleLayer
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/gpt3'

    C.data = CN()
    C.data.block_size = 2048

    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt3-1l'

    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    config.model.block_size = config.data.block_size = 2048


    device = torch.device(params.devname)
    dtype = torch.float16
    model = GPT3SingleLayer(config.model).to(device).to(dtype)

    # setup the optimizer
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()

    x = torch.randn((params.bs, 2048, config.model.n_embd), device=device, dtype=dtype)


    def roi():
        loss = model(x).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


    benchmark_wrapper('gpt3', roi)
