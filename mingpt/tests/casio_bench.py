import os
import sys

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import params
import cudaprofile
from torch_wrapper import benchmark_wrapper

from mingpt.model import GPT

from mingpt.model import GPT
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

    C.data = FakeDataset.get_default_config()

    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt3-1l'

    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class FakeDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 2048
        return C

    def __init__(self, config):
        self.config = config
        self.vocab_size = 1

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = torch.randint(0, 1, (2048,), dtype=torch.long)
        y = torch.randint(0, 1, (2048,), dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    train_dataset = FakeDataset(config.data)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()

    device = torch.device(params.devname)
    dtype = torch.float16

    model = GPT(config.model).to(device).to(dtype)

    # setup the optimizer
    optimizer = model.configure_optimizers(config.trainer)

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=params.bs,
        num_workers=0,
    )


    model.train()

    data_iter = iter(train_loader)

    batch = next(data_iter)
    batch = [t.to(device) for t in batch]

    def roi():
        x, y = batch
        logits, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)
        optimizer.step()


    benchmark_wrapper('gpt3', roi)
