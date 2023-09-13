import torch

from utils import params
from utils.torch_wrapper import benchmark_wrapper2

from .model import *

device = torch.device(params.devname)
cfg = bert_large_conf(512)
bert = Bert(cfg)
bert.to(params.dtype_torch).to(device)

input_ids = torch.randint(0, cfg.vocab_size, (params.bs, 512)).to(device)
token_type_ids = torch.randint(0, 1, (params.bs, 512)).to(device)

def roi():
    _ = bert(input_ids, token_type_ids)

benchmark_wrapper2(roi)
