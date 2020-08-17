import numpy as np
import torch
from torch import nn

def gcd(a,b):
    a, b = (a, b) if a >=b else (b, a)
    if a%b == 0:
        return b
    else :
        return gcd(b,a%b)

def lcm(a,b):
    return a*b//gcd(a,b)


if __name__ == "__main__":
    print(lcm(3,2))

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths.device)
    #print ids
    mask = (ids < lengths.unsqueeze(1))
    return mask

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def test_mask():
    lengths = torch.IntTensor([3,5,4])
    print(torch.ceil(lengths.float() / 2))

    data = torch.FloatTensor(3, 5, 2) # [B, T, D]
    data.fill_(1.)
    m = get_mask_from_lengths(lengths.cuda(), data.size(1))
    print(m)
    m =  m.unsqueeze(2).expand(-1,-1,data.size(2)).float()
    print(m)

    print(torch.sum(data.cuda() * m) / torch.sum(m))


def test_loss():
    data1 = torch.FloatTensor(3, 5, 2)
    data1.fill_(1.)
    data2 = torch.FloatTensor(3, 5, 2)
    data2.fill_(2.)
    data2[0,0,0] = 1000

    l = torch.nn.L1Loss(reduction='none')(data1,data2)
    print(l)


def initialize(model):
    print("initialize", model.__class__.__name__)
    initrange = 0.01
    bias_initrange = 0.001
    parameters = model.parameters()
    for param in parameters:
        if len(param.shape) >= 2:
            torch.nn.init.xavier_normal_(param)
        else:
            torch.nn.init.uniform_(param, (-1)*bias_initrange, bias_initrange)

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-initrange, initrange)

#if __name__ == '__main__':
#    test_mask()
