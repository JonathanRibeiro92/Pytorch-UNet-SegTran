import numpy
import random
import torch


def set_all_lib_seed(seed: int, loader=None) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

    return None


def set_seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    return None
