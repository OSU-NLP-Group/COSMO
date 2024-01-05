import sys
import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re

from torch.utils.data import DataLoader

try:
    from torch_geometric.data import Batch
except ImportError:
    pass

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")
