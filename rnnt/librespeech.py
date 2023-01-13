

#  {
#     "transcript": "if the reader will excuse me i will say nothing of my antecedents nor of the circumstances which led me to leave my native country the narrative would be tedious to him and painful to myself",
#     "files": [
#       {
#         "channels": 1,
#         "sample_rate": 16000.0,
#         "bitdepth": 16,
#         "bitrate": 256000.0,
#         "duration": 11.66,
#         "num_samples": 186560,
#         "encoding": "Signed Integer PCM",
#         "silent": false,
#         "fname": "dev-clean-wav/2412/153948/2412-153948-0000.wav",
#         "speed": 1
#       }
#     ],
#     "original_duration": 11.66,
#     "original_num_samples": 186560
#   },

import os
import json
import torch
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf

class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=None, trim=False, trim_db=60):
        samples = self._convert_samples_to_float32(samples)

        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)

        self.samples = samples
        self.sample_rate = sample_rate
        if self.samples.ndim >= 2:
            self.samples = np.mean(self.samples, 1)

    def __eq__(self, other):
        if type(other) is not type(self): return False
        if self.sample_rate != other.sample_rate: return False
        if self.samples.shape != other.samples.shape: return False
        if np.any(self.samples != other.samples): return False
        return True

    def __ne__(self, other): return not self.__eq__(other)

    def __repr__(self): return \
        f'AudioSegment[{self.num_samples} samples, {self.sample_rate} samp/sec, duration={self.duration} sec, rms={self.rms_db} dB]'

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']: pass
        else: raise TypeError(f'Unsupported sample type: {samples.dtype}.')
        return float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=None, int_values=False, offset=0, duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype = 'int32' if int_values else 'float32'
            sample_rate = f.samplerate
            if offset > 0: f.seek(int(offset * sample_rate))
            if duration > 0: samples = f.read(int(duration * sample_rate), dtype=dtype)
            else: samples = f.read(dtype=dtype)
        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def num_samples(self): return self.samples.shape[0]

    @property
    def duration(self): return self.samples.shape[0] / float(self.sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self.samples ** 2)
        return 10 * np.log10(mean_square)

    def gain_db(self, gain): self.samples *= 10. ** (gain / 20.)

    def pad(self, pad_size, symmetric=False):
        """Add zero padding to the sample. The pad size is given in number of samples.
        If symmetric=True, `pad_size` will be added to both sides. If false, `pad_size`
        zeros will be added only to the end.
        """
        self.samples = np.pad(
            self.samples,
            (pad_size if symmetric else 0, pad_size),
            mode='constant')

    def subsegment(self, start_time=None, end_time=None):
        """Cut the AudioSegment between given boundaries.
        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set, e.g.
            out of bounds in time.
        """
        start_time = 0.0 if start_time is None else start_time
        end_time = self.duration if end_time is None else end_time

        if start_time < 0.0:  start_time = self.duration + start_time
        if end_time < 0.0: end_time = self.duration + end_time

        if start_time < 0.0: raise ValueError(
            f'The slice start position ({start_time} s) is out of bounds.')
        if end_time < 0.0: raise ValueError(
            f'The slice end position ({end_time} s) is out of bounds.')

        if start_time > end_time: raise ValueError(
            f'The slice start position ({start_time} s) is later than the end position ({end_time} s).')

        if end_time > self.duration: raise ValueError(
            f'The slice end position ({end_time} s) is out of bounds (> {self.duration} s)')

        start_sample = int(round(start_time * self.sample_rate))
        end_sample = int(round(end_time * self.sample_rate))
        self.samples = self.samples[start_sample:end_sample]


@dataclass
class LibrespeechSample:
    meta : dict
    transcript: str
    audio : AudioSegment

class Librespeech(torch.utils.data.Dataset):
    def __init__(self, manifest_file):
        self.basedir = os.path.dirname(manifest_file)
        self.meta = json.load(open(manifest_file, 'r'))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if type(idx) is int:
            sample_id = idx
            file_id = 0
            assert len(self.meta[sample_id]['files']) == 1
        elif type(idx) is tuple:
            sample_id = idx[0]
            file_id = idx[1]
        meta = self.meta[sample_id]['files'][file_id]
        transcript = self.meta[sample_id]['transcript']
        fname = self.meta[sample_id]['files'][file_id]['fname']
        return LibrespeechSample(
            meta,
            transcript,
            AudioSegment.from_file(f'{self.basedir}/{fname}'))
