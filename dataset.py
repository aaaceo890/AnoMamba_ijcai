import logging
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data

class RecDataset(IterableDataset):
    def __init__(self, data, label, window_length=-1, dtype=np.float32,
                 partition=False, align="none", normalization_type="norm",
                 xscaler=None, output_dim=None, shuffle=True):
        super(RecDataset, self).__init__()
        data = np.nan_to_num(data, 0)
        # T, D
        self.data = data.astype(dtype)
        # normalize
        if xscaler is not None:
            self.data = xscaler.transform(self.data)
        else:
            if normalization_type == "norm":
                xscaler = StandardScaler()
                self.data = xscaler.fit_transform(self.data)
            elif normalization_type == "min-max":
                xscaler = MinMaxScaler()
                self.data = xscaler.fit_transform(self.data)
            else:
                xscaler = None
        self.xscaler = xscaler
        # T, 1
        # 0: normal, 1: outlier
        label = label.astype(np.int32)
        if window_length == -1:
            window_length = self.data.shape[0]
        self.window_length = window_length
        if len(self.data) < self.window_length:
            logging.warning(
                "set window length to {}, but original data length is {}, automatically fit window length to {}".format(
                    self.window_length, len(self.data), len(self.data)
                ))
            self.window_length = len(self.data)

        # specific dimension
        self.input_dim = self.data.shape[-1]
        if output_dim is None:
            self.output_dim = self.input_dim
        else:
            self.output_dim = output_dim

        self.sampler = DistributedSampler(partition=partition, shuffle=shuffle)
        self.align = align
        self.label = label

    def __len__(self):
        if self.align == "causal_pad":
            return len(self.data)
        else:
            return len(self.data) - (self.window_length - 1)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        indexes = self.sampler.sample(self.data[:self.__len__()])
        for index in indexes:
            if self.align == "causal_pad":
                start_index = max(0, index - self.window_length + 1)
                pad_len = max(0, self.window_length - index - 1)
                end_index = index
                x = np.concatenate([np.zeros((pad_len, self.data.shape[-1])), self.data[start_index:end_index+1]],
                                   dtype=self.data.dtype, axis=0)
                y = self.data[index:index+1]

                yield x, y
            else:
                # window data
                end_index = min(index + self.window_length, len(self.data))
                x = self.data[index: end_index]
                y = self.data[index: end_index, :self.output_dim]
                yield x, y