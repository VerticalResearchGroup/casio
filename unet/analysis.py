
import torch
import torch.fx
import model.unet3d as U

model = U.Unet3D(1, 3, normalization='instancenorm', activation='relu')
print(model)

x = torch.rand(1, 1, 128, 128, 128)

model(x)
