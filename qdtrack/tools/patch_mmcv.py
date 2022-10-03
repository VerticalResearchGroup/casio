import mmcv
import mmcv.parallel.data_container
import mmcv.runner.epoch_based_runner as X
import time
import torch

import sys
import os

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile
import params
from torch_wrapper import benchmark_wrapper

dev = torch.device(params.devname)

def convert_tensors(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float32:
            x = x.half()
        return x.to(dev)
    elif isinstance(x, list):
        return [convert_tensors(y) for y in x]
    elif isinstance(x, tuple):
        return tuple(convert_tensors(y) for y in x)
    elif isinstance(x, mmcv.parallel.data_container.DataContainer):
        return convert_tensors(x.data)
    elif isinstance(x, dict):
        return {k: convert_tensors(v) for k, v in x.items()}
    else:
        return x

def train(self, data_loader, **kwargs):
    self.model.train()
    print(self)

    self.mode = 'train'
    self.data_loader = data_loader
    self._max_iters = self._max_epochs * len(self.data_loader)
    self.call_hook('before_train_epoch')
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    for i, data_batch in enumerate(self.data_loader):
        self.data_batch = data_batch
        self._inner_iter = i

        print(type(data_batch))

        data_batch = convert_tensors(data_batch)

        for k in data_batch:
            data_batch[k] = data_batch[k][0]

        # for k, v in data_batch.items():
        #     # print(k, type(v.data))
        #     if isinstance(v, list):
        #         for x in v.data:
        #             if isinstance(x, torch.Tensor):
        #                 x.half()

        def roi():
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')

        benchmark_wrapper('qdtrack', roi)
        exit(0)

        # torch.cuda.synchronize()
        # cudaprofile.start()

        # t0 = time.perf_counter()

        # for j in range(30):

        # del self.data_batch
        # self._iter += 1

        # torch.cuda.synchronize()
        # t1 = time.perf_counter()
        # cudaprofile.end()

        # cuda_total = start.elapsed_time(end)

        # print(f'Total Time: {t1 - t0}')
        # #print(f'Throughput: {args.num_iters * args.batch / (t1 - t0)}')
        # print()
        # print(f'(CUDA) Total Time: {cuda_total/1000}')
        # #print(f'(CUDA) Throughput: {args.num_iters * args.batch / (cuda_total/1000)}')

        break

    self.call_hook('after_train_epoch')
    self._epoch += 1

print('Monkey patching EpochBasedRunner.train...')
X.EpochBasedRunner.train = train
