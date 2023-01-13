
import torch
import numpy as np

import rnnt

def make_torch_tensor(name, data):
    if 'weight' in name or 'bias' in name:
        return torch.nn.Parameter(torch.tensor(data))
    else: return torch.tensor(data)

def rnnt_load_from_weights(self : rnnt.Rnnt, weights):
    self.encoder.pre_lstms.lstm.weight_ih_l0 = weights['encoder.pre_rnn.lstm.weight_ih_l0']
    self.encoder.pre_lstms.lstm.weight_hh_l0 = weights['encoder.pre_rnn.lstm.weight_hh_l0']
    self.encoder.pre_lstms.lstm.bias_ih_l0 = weights['encoder.pre_rnn.lstm.bias_ih_l0']
    self.encoder.pre_lstms.lstm.bias_hh_l0 = weights['encoder.pre_rnn.lstm.bias_hh_l0']
    self.encoder.pre_lstms.lstm.weight_ih_l1 = weights['encoder.pre_rnn.lstm.weight_ih_l1']
    self.encoder.pre_lstms.lstm.weight_hh_l1 = weights['encoder.pre_rnn.lstm.weight_hh_l1']
    self.encoder.pre_lstms.lstm.bias_ih_l1 = weights['encoder.pre_rnn.lstm.bias_ih_l1']
    self.encoder.pre_lstms.lstm.bias_hh_l1 = weights['encoder.pre_rnn.lstm.bias_hh_l1']
    self.encoder.post_lstms.lstm.weight_ih_l0 = weights['encoder.post_rnn.lstm.weight_ih_l0']
    self.encoder.post_lstms.lstm.weight_hh_l0 = weights['encoder.post_rnn.lstm.weight_hh_l0']
    self.encoder.post_lstms.lstm.bias_ih_l0 = weights['encoder.post_rnn.lstm.bias_ih_l0']
    self.encoder.post_lstms.lstm.bias_hh_l0 = weights['encoder.post_rnn.lstm.bias_hh_l0']
    self.encoder.post_lstms.lstm.weight_ih_l1 = weights['encoder.post_rnn.lstm.weight_ih_l1']
    self.encoder.post_lstms.lstm.weight_hh_l1 = weights['encoder.post_rnn.lstm.weight_hh_l1']
    self.encoder.post_lstms.lstm.bias_ih_l1 = weights['encoder.post_rnn.lstm.bias_ih_l1']
    self.encoder.post_lstms.lstm.bias_hh_l1 = weights['encoder.post_rnn.lstm.bias_hh_l1']
    self.encoder.post_lstms.lstm.weight_ih_l2 = weights['encoder.post_rnn.lstm.weight_ih_l2']
    self.encoder.post_lstms.lstm.weight_hh_l2 = weights['encoder.post_rnn.lstm.weight_hh_l2']
    self.encoder.post_lstms.lstm.bias_ih_l2 = weights['encoder.post_rnn.lstm.bias_ih_l2']
    self.encoder.post_lstms.lstm.bias_hh_l2 = weights['encoder.post_rnn.lstm.bias_hh_l2']
    self.prediction.embed.weight = weights['prediction.embed.weight']
    self.prediction.dec_rnn.lstm.weight_ih_l0 = weights['prediction.dec_rnn.lstm.weight_ih_l0']
    self.prediction.dec_rnn.lstm.weight_hh_l0 = weights['prediction.dec_rnn.lstm.weight_hh_l0']
    self.prediction.dec_rnn.lstm.bias_ih_l0 = weights['prediction.dec_rnn.lstm.bias_ih_l0']
    self.prediction.dec_rnn.lstm.bias_hh_l0 = weights['prediction.dec_rnn.lstm.bias_hh_l0']
    self.prediction.dec_rnn.lstm.weight_ih_l1 = weights['prediction.dec_rnn.lstm.weight_ih_l1']
    self.prediction.dec_rnn.lstm.weight_hh_l1 = weights['prediction.dec_rnn.lstm.weight_hh_l1']
    self.prediction.dec_rnn.lstm.bias_ih_l1 = weights['prediction.dec_rnn.lstm.bias_ih_l1']
    self.prediction.dec_rnn.lstm.bias_hh_l1 = weights['prediction.dec_rnn.lstm.bias_hh_l1']
    self.joint.net[0].weight = weights['joint.net.0.weight']
    self.joint.net[0].bias = weights['joint.net.0.bias']
    self.joint.net[3].weight = weights['joint.net.3.weight']
    self.joint.net[3].bias = weights['joint.net.3.bias']


rnnt.Rnnt.load_from_weights = rnnt_load_from_weights

def rnntmodel_load_from_file(self : rnnt.Rnnt, filename):
    weights = {
        name: make_torch_tensor(name, data)
        for name, data in np.load(filename).items()
    }
    self.load_from_weights(weights)

rnnt.Rnnt.load_from_file = rnntmodel_load_from_file
