import torch


def GetDevice():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')