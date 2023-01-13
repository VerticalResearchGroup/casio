import torch
import rnnt
import librespeech

rnnt = rnnt.Rnnt()
rnnt.eval()
rnnt.load_from_file('/research/data/mlmodels/npz/rnnt.npz')

dataset = librespeech.Librespeech('./librespeech-min/librespeech-min.json')

x = torch.Tensor(dataset[0].audio.samples)[:int(239*186560/389)].unsqueeze_(0)
l = torch.LongTensor([int(239*186560/389)])

rnnt(x, l)

