import numpy as np
import torch


def worker_init_fn_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    seed = worker_seed + worker_id
    np.random.seed(seed)