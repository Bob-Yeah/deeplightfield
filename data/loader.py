import torch
import math
from my import device


class FastDataLoader(object):

    class Iter(object):

        def __init__(self, dataset, batch_size, shuffle, drop_last) -> None:
            super().__init__()
            self.indices = torch.randperm(len(dataset), device=device.GetDevice()) \
                if shuffle else torch.arange(len(dataset), device=device.GetDevice())
            self.offset = 0
            self.batch_size = batch_size
            self.dataset = dataset
            self.drop_last = drop_last

        def __next__(self):
            if self.offset + (self.batch_size if self.drop_last else 0) >= len(self.dataset):
                raise StopIteration()
            indices = self.indices[self.offset:self.offset + self.batch_size]
            self.offset += self.batch_size
            return self.dataset[indices]

    def __init__(self, dataset, batch_size, shuffle, drop_last, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        return FastDataLoader.Iter(self.dataset, self.batch_size,
                                   self.shuffle, self.drop_last)

    def __len__(self):
        return math.floor(len(self.dataset) / self.batch_size) if self.drop_last \
            else math.ceil(len(self.dataset) / self.batch_size)
