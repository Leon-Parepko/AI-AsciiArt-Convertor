
import torch


def gpu_check():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')