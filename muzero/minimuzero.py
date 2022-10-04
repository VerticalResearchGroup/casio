import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')

import cudaprofile
from torch_wrapper import benchmark_wrapper
import params


# import nevergrad
import numpy
# import ray
import torch
from torch.utils.tensorboard import SummaryWriter

# import diagnose_model
import models
# import replay_buffer
# import self_play
# import shared_storage

game_name = sys.argv[1]

try:
    game_module = importlib.import_module("games." + game_name)
    Game = game_module.Game
    config = game_module.MuZeroConfig()
except ModuleNotFoundError as err:
    print( f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.')
    raise err


if game_name == 'tictactoe':
    # BS = 1 .. 64
    batch_size = params.bs
    rnn_length = config.num_unroll_steps
elif game_name == 'atari':
    # BS = 1 .. 1024
    batch_size = params.bs
    rnn_length = config.num_unroll_steps

device = torch.device(params.devname)
model = models.MuZeroNetwork(config)
assert torch.cuda.is_available()
model = model.half()

model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if game_name == 'tictactoe':
    observation_batch = \
        torch.rand(batch_size, 3, 3, 3, dtype=torch.float16, device=device)
elif game_name == 'atari':
    observation_batch = \
        torch.rand(batch_size, 131, 96, 96, dtype=torch.float16, device=device)

action_batch = torch.rand(batch_size, 1, dtype=torch.float16, device=device)


def roi():
    # st = time.time()
    opt.zero_grad()
    value, reward, policy_logits, hidden_state = model.initial_inference(observation_batch)
    loss = value.sum() + reward.sum() + policy_logits.sum()
    for i in range(1, rnn_length):
        value, reward, policy_logits, hidden_state = model.recurrent_inference( hidden_state, action_batch)
        loss += value.sum() + reward.sum() + policy_logits.sum()
    # print("fwd ", time.time() - st)

    # st = time.time()
    loss.backward()
    opt.step()
    # print("bwd ", time.time() - st)

benchmark_wrapper('muzero', roi)
