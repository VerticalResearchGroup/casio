import torch
import numpy as np

def monkey_patch_nn_mod(mod):
    print(f'Patching {mod.__name__}.forward')
    fwd = mod.forward
    def decorator(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs, orig=fwd)

        mod.forward = wrapper
        return wrapper
    return decorator

def monky_patch_torch_func(func):
    print(f'Patching torch.{func.__name__}')
    def decorator(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs, orig=func)

        setattr(torch, func.__name__, wrapper)
        return wrapper
    return decorator

with open('gemms.txt', 'w') as f: pass

@monkey_patch_nn_mod(torch.nn.Conv2d)
def conv2d(self, input, orig=None):
    out = orig(self, input)

    B, C, H, W = input.shape
    K, _C, R, S = self.weight.shape
    stride = self.stride[0]
    _B, _K, P, Q = out.shape

    assert _B == B
    assert _K == K
    assert _C == C

    with open('gemms.txt', 'a') as f:
        print(f'Conv2D({B}, {C}, {K}, {H}, {W}, {P}, {Q}, {R}, {S}, {stride})', file=f)
    return out

@monkey_patch_nn_mod(torch.nn.Linear)
def linear(self, input, orig=None):
    out = orig(self, input)

    M = np.prod(input.shape[:-1])
    K = input.shape[-1]
    N, _K = self.weight.shape

    assert _K == K

    with open('gemms.txt', 'a') as f:
        print(f'Linear({M}, {N}, {K})', file=f)

    return out

@monky_patch_torch_func(torch.matmul)
def matmul(input, other, orig=None):
    out = orig(input, other)

    # print(input.shape, other.shape)

    M = np.prod(input.shape[:-1])
    K = input.shape[-1]
    N = other.shape[-1]


    with open('gemms.txt', 'a') as f:
        print(f'Matmul({M}, {N}, {K})', file=f)

    return out

@monky_patch_torch_func(torch.bmm)
def bmm(input, mat2, orig=None):
    out = orig(input, mat2)

    L, M, K = input.shape
    L, _K, N = mat2.shape

    assert _K == K

    with open('gemms.txt', 'a') as f:
        print(f'BatchMatmul({L}, {M}, {N}, {K})', file=f)
    return out
