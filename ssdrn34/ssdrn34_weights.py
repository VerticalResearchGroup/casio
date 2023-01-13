
import torch
import numpy as np

import ssdrn34

def ssdrn34_load_weights(self : ssdrn34.SsdRn34_1200, filename):
    def make_torch_tensor(name, data):
        if 'weight' in name or 'bias' in name:
            return torch.nn.Parameter(torch.tensor(data))
        else: return torch.tensor(data)

    weights = {
        name: make_torch_tensor(name, data)
        for name, data in np.load(filename).items()
    }

    self.rn34[0].weight = weights['model.layer1.0.weight']
    self.rn34[1].weight = weights['model.layer1.1.weight']
    self.rn34[1].bias = weights['model.layer1.1.bias']
    self.rn34[1].running_mean = weights['model.layer1.1.running_mean']
    self.rn34[1].running_var = weights['model.layer1.1.running_var']
    self.rn34[1].num_batches_tracked = weights['model.layer1.1.num_batches_tracked']

    def map_resnet_block(dest, src_name_base, downsample=False):
        dest.conv1.weight = weights[f'{src_name_base}.conv1.weight']

        dest.bn1.weight = weights[f'{src_name_base}.bn1.weight']
        dest.bn1.bias = weights[f'{src_name_base}.bn1.bias']
        dest.bn1.running_mean = weights[f'{src_name_base}.bn1.running_mean']
        dest.bn1.running_var = weights[f'{src_name_base}.bn1.running_var']
        dest.bn1.num_batches_tracked = weights[f'{src_name_base}.bn1.num_batches_tracked']

        dest.conv2.weight = weights[f'{src_name_base}.conv2.weight']

        dest.bn2.weight = weights[f'{src_name_base}.bn2.weight']
        dest.bn2.bias = weights[f'{src_name_base}.bn2.bias']
        dest.bn2.running_mean = weights[f'{src_name_base}.bn2.running_mean']
        dest.bn2.running_var = weights[f'{src_name_base}.bn2.running_var']
        dest.bn2.num_batches_tracked = weights[f'{src_name_base}.bn2.num_batches_tracked']

        if downsample:
            dest.downsample[0].weight = weights[f'{src_name_base}.downsample.0.weight']
            dest.downsample[1].weight = weights[f'{src_name_base}.downsample.1.weight']
            dest.downsample[1].bias = weights[f'{src_name_base}.downsample.1.bias']
            dest.downsample[1].running_mean = weights[f'{src_name_base}.downsample.1.running_mean']
            dest.downsample[1].running_var = weights[f'{src_name_base}.downsample.1.running_var']
            dest.downsample[1].num_batches_tracked = weights[f'{src_name_base}.downsample.1.num_batches_tracked']

    map_resnet_block(self.rn34[4][0], 'model.layer1.4.0')
    map_resnet_block(self.rn34[4][1], 'model.layer1.4.1')
    map_resnet_block(self.rn34[4][2], 'model.layer1.4.2')

    map_resnet_block(self.rn34[5][0], 'model.layer1.5.0', downsample=True)
    map_resnet_block(self.rn34[5][1], 'model.layer1.5.1')
    map_resnet_block(self.rn34[5][2], 'model.layer1.5.2')
    map_resnet_block(self.rn34[5][3], 'model.layer1.5.3')

    map_resnet_block(self.rn34[6][0], 'model.layer2.0.0', downsample=True)
    map_resnet_block(self.rn34[6][1], 'model.layer2.0.1')
    map_resnet_block(self.rn34[6][2], 'model.layer2.0.2')
    map_resnet_block(self.rn34[6][3], 'model.layer2.0.3')
    map_resnet_block(self.rn34[6][4], 'model.layer2.0.4')
    map_resnet_block(self.rn34[6][5], 'model.layer2.0.5')

    def map_additional_block(dest, src_name_base):
        dest[0].weight = weights[f'{src_name_base}.0.weight']
        dest[0].bias = weights[f'{src_name_base}.0.bias']
        dest[2].weight = weights[f'{src_name_base}.2.weight']
        dest[2].bias = weights[f'{src_name_base}.2.bias']

    map_additional_block(self.blocks[1], 'additional_blocks.0')
    map_additional_block(self.blocks[2], 'additional_blocks.1')
    map_additional_block(self.blocks[3], 'additional_blocks.2')
    map_additional_block(self.blocks[4], 'additional_blocks.3')
    map_additional_block(self.blocks[5], 'additional_blocks.4')

    for i in range(6):
        self.loc[i].weight = weights[f'loc.{i}.weight']
        self.loc[i].bias = weights[f'loc.{i}.bias']
        self.conf[i].weight = weights[f'conf.{i}.weight']
        self.conf[i].bias = weights[f'conf.{i}.bias']


ssdrn34.SsdRn34_1200.load_weights = ssdrn34_load_weights

