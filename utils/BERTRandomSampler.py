import torch.utils.data.sampler as sampler
import numpy as np
import time

class BERTRandomSampler(sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = np.arange(len(self.data_source))
        np.random.seed(int(round(time.time() % 100000)))
        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)