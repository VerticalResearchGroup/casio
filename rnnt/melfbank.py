# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import librosa

class Normalize(Enum):
    PER_FEATURE = 0
    ALL_FEATURES = 1

class MelWindow(Enum):
    HANN = 0
    HAMMING = 1
    BLACKMAN = 2
    BARTLETT = 3

class MelFilterBanks(nn.Module):
    def __init__(
        self, /,
        sample_rate : int,
        window_size : float,
        window_stride : float,
        max_duration : float,
        lowfreq : int,
        highfreq : Optional[int],
        preemph : Optional[float],
        normalize : Normalize,
        nfeatures : int,
        frame_splicing : int,
        dither : float,
        log : bool,
        window : Optional[MelWindow],
        nfft : Optional[int],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_duration = max_duration
        self.lowfreq = lowfreq
        self.highfreq = highfreq if highfreq is not None else self.sample_rate / 2
        self.preemph = preemph
        self.normalize = normalize
        self.nfeatures = nfeatures
        self.frame_splicing = frame_splicing
        self.dither = dither
        self.log = log
        self.deterministic_dither = True

        self.window_samples = int(self.sample_rate * self.window_size)
        self.stride_samples = int(self.sample_rate * self.window_stride)
        self.nfft = nfft if nfft is not None else 2 ** np.ceil(np.log2(self.window_samples))

        if window is not None:
            window_fn = {
                MelWindow.HANN: torch.hann_window,
                MelWindow.HAMMING: torch.hamming_window,
                MelWindow.BLACKMAN: torch.blackman_window,
                MelWindow.BARTLETT: torch.bartlett_window
            }[window]
            self.register_buffer('window_tensor',
                window_fn(self.window_samples, periodic=False))
        else:
            self.register_buffer('window_tensor', None)

        self.window_tensor : Optional[torch.Tensor]

        self.register_buffer('fb', torch.tensor(
            librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.nfft,
                n_mels=self.nfeatures,
                fmin=self.lowfreq,
                fmax=self.highfreq),
            dtype=torch.float).unsqueeze(0))

        self.fb : torch.Tensor

        max_length = 1 + np.ceil((self.max_duration * self.sample_rate - self.window_samples) / self.stride_samples)
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad


    def get_seq_len(self, seq_len):
        seq_len = (seq_len + self.stride_samples - 1) // self.stride_samples
        seq_len = (seq_len + self.frame_splicing - 1) // self.frame_splicing
        return seq_len

    @torch.no_grad()
    def forward(self, x, seq_len):
        dtype = x.dtype
        seq_len = self.get_seq_len(seq_len)

        if self.dither > 0 and not self.deterministic_dither:
            x += self.dither * torch.randn_like(x)

        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                dim=1)

        x = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.stride_samples,
            win_length=self.window_samples,
            center=True,
            window=self.window_tensor.to(dtype=torch.float))

        # get power spectrum
        x = x.pow(2).sum(-1)

        if self.dither > 0 and self.deterministic_dither:
            x = x + self.dither ** 2

        x = torch.matmul(self.fb.to(x.dtype), x)

        if self.log: x = torch.log(x + 1e-20)

        if self.frame_splicing > 1:
            seq = [x]
            for n in range(1, self.frame_splicing):
                tmp = torch.zeros_like(x)
                tmp[:, :, :-n] = x[:, :, n:]
                seq.append(tmp)
            x = torch.cat(seq, dim=1)[:, :, ::self.frame_splicing]

        constant = 1e-5
        if self.normalize == Normalize.PER_FEATURE:
            x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
                # make sure x_std is not zero
                x_std += constant
            x = (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

        elif self.normalize == Normalize.ALL_FEATURES:
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, :seq_len[i].item()].mean()
                x_std[i] = x[i, :, :seq_len[i].item()].std()
                # make sure x_std is not zero
                x_std += constant
            x = (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)

        return x[:, :, :seq_len.max()].to(dtype), seq_len

    @staticmethod
    def rnnt_mel_filter_banks():
        return MelFilterBanks(
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            max_duration=16.7,
            lowfreq=0,
            highfreq=None,
            normalize=Normalize.PER_FEATURE,
            nfeatures=80,
            nfft=512,
            preemph=0.97,
            frame_splicing=3,
            dither=0.00001,
            log=True,
            window=MelWindow.HANN)
