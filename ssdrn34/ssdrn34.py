import torch
import numpy as np
import multibox

from torchvision.models.resnet import resnet34

def _patch_rn34_layer(layer : torch.nn.Module):
    layer[0].conv1.stride = (1, 1)
    layer[0].downsample[0].stride = (1, 1)
    layer[1].conv1.stride = (1, 1)
    layer[2].conv1.stride = (1, 1)
    layer[3].conv1.stride = (1, 1)
    layer[4].conv1.stride = (1, 1)
    layer[5].conv1.stride = (1, 1)
    return layer

def _patch_rn34_for_ssd(rn34 : torch.nn.Module):
    return torch.nn.Sequential(*[
        rn34.conv1,
        rn34.bn1,
        rn34.relu,
        rn34.maxpool,
        rn34.layer1,
        rn34.layer2,
        _patch_rn34_layer(rn34.layer3)
    ])

class SsdRn34_1200(torch.nn.Module):
    num_classes = 81
    num_dboxes = [4, 6, 6, 6, 4, 4]
    chans = [256, 512, 512, 256, 256, 256]
    strides = [3, 3, 2, 2, 2, 2]

    @staticmethod
    def extra_block(
        in_chan : int,
        out_chan : int,
        int_chan : int,
        ksize1 : int,
        ksize2 : int,
        stride2 : int,
        pad2 : int = 1
    ):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_chan, int_chan, kernel_size=ksize1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int_chan, out_chan, kernel_size=ksize2, padding=pad2, stride=stride2),
            torch.nn.ReLU(inplace=True)
        )

    @staticmethod
    def make_loc():
        return torch.nn.ModuleList([
            torch.nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1, stride=SsdRn34_1200.strides[0])
            for nd, oc in zip(SsdRn34_1200.num_dboxes, SsdRn34_1200.chans)
        ])

    @staticmethod
    def make_conf():
        return torch.nn.ModuleList([
            torch.nn.Conv2d(oc, nd * SsdRn34_1200.num_classes, kernel_size=3, padding=1, stride=SsdRn34_1200.strides[1])
            for nd, oc in zip(SsdRn34_1200.num_dboxes, SsdRn34_1200.chans)
        ])

    def __init__(self):
        super().__init__()
        self.rn34 = _patch_rn34_for_ssd(resnet34())
        self.blocks = torch.nn.ModuleList([
            self.rn34,
            SsdRn34_1200.extra_block(SsdRn34_1200.chans[0], SsdRn34_1200.chans[1], 256, 1, 3, SsdRn34_1200.strides[2], 1),
            SsdRn34_1200.extra_block(SsdRn34_1200.chans[1], SsdRn34_1200.chans[2], 256, 1, 3, SsdRn34_1200.strides[3], 1),
            SsdRn34_1200.extra_block(SsdRn34_1200.chans[2], SsdRn34_1200.chans[3], 256, 1, 3, SsdRn34_1200.strides[4], 1),
            SsdRn34_1200.extra_block(SsdRn34_1200.chans[3], SsdRn34_1200.chans[4], 256, 1, 3, SsdRn34_1200.strides[5], 0),
            SsdRn34_1200.extra_block(SsdRn34_1200.chans[4], SsdRn34_1200.chans[5], 256, 1, 3, 1, 0)
        ])
        self.loc = SsdRn34_1200.make_loc()
        self.conf = SsdRn34_1200.make_conf()
        self.encoder = multibox.Encoder(
            multibox.dboxes_R34_coco([1200, 1200], SsdRn34_1200.strides))

    def forward(self, x):
        activations = []
        for l in self.blocks:
            x = l(x)
            activations.append(x)

        locs = []
        confs = []
        for act, loc, conf in zip(activations, self.loc, self.conf):
            nbatch = act.size(0)
            locs.append(loc(act).view(nbatch, 4, -1))
            confs.append(conf(act).view(nbatch, SsdRn34_1200.num_classes, -1))

        return [torch.concat(locs, 2), torch.concat(confs, 2)]
        return self.encoder.decode_batch(
            torch.concat(locs, 2), torch.concat(confs, 2), 0.50, 200)


class SsdRn34_300(torch.nn.Module):
    num_classes = 81
    num_dboxes = [4, 6, 6, 6, 4, 4]
    chans = [256, 512, 512, 256, 256, 256]
    strides = [3, 3, 2, 2, 2, 1, 1]

    @staticmethod
    def extra_block(
        in_chan : int,
        out_chan : int,
        int_chan : int,
        ksize1 : int,
        ksize2 : int,
        stride2 : int,
        pad2 : int = 1
    ):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_chan, int_chan, kernel_size=ksize1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int_chan, out_chan, kernel_size=ksize2, padding=pad2, stride=stride2),
            torch.nn.ReLU(inplace=True)
        )

    @staticmethod
    def make_loc():
        return torch.nn.ModuleList([
            torch.nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1, stride=SsdRn34_300.strides[0])
            for nd, oc in zip(SsdRn34_300.num_dboxes, SsdRn34_300.chans)
        ])

    @staticmethod
    def make_conf():
        return torch.nn.ModuleList([
            torch.nn.Conv2d(oc, nd * SsdRn34_300.num_classes, kernel_size=3, padding=1, stride=SsdRn34_300.strides[1])
            for nd, oc in zip(SsdRn34_300.num_dboxes, SsdRn34_300.chans)
        ])

    def __init__(self):
        super().__init__()
        self.rn34 = _patch_rn34_for_ssd(resnet34())
        self.blocks = torch.nn.ModuleList([
            self.rn34,
            SsdRn34_300.extra_block(SsdRn34_300.chans[0], SsdRn34_300.chans[1], 256, 1, 3, SsdRn34_300.strides[2], 1),
            SsdRn34_300.extra_block(SsdRn34_300.chans[1], SsdRn34_300.chans[2], 256, 1, 3, SsdRn34_300.strides[3], 1),
            SsdRn34_300.extra_block(SsdRn34_300.chans[2], SsdRn34_300.chans[3], 128, 1, 3, SsdRn34_300.strides[4], 1),
            SsdRn34_300.extra_block(SsdRn34_300.chans[3], SsdRn34_300.chans[4], 128, 1, 3, SsdRn34_300.strides[5], 0),
            SsdRn34_300.extra_block(SsdRn34_300.chans[4], SsdRn34_300.chans[5], 128, 1, 3, SsdRn34_300.strides[6], 0)
        ])
        self.loc = SsdRn34_300.make_loc()
        self.conf = SsdRn34_300.make_conf()
        self.encoder = multibox.Encoder(
            multibox.dboxes_R34_coco([1200, 1200], SsdRn34_300.strides))

    def forward(self, x):
        activations = []
        for l in self.blocks:
            x = l(x)
            activations.append(x)

        locs = []
        confs = []
        for act, loc, conf in zip(activations, self.loc, self.conf):
            nbatch = act.size(0)
            locs.append(loc(act).view(nbatch, 4, -1))
            confs.append(conf(act).view(nbatch, SsdRn34_300.num_classes, -1))

        return [torch.concat(locs, 2), torch.concat(confs, 2)]
        return self.encoder.decode_batch(
            torch.concat(locs, 2), torch.concat(confs, 2), 0.50, 200)
