import torch
import torch.nn as nn

def accuracy(outs, targets):
    _, preds = torch.max(outs.data, -1)
    return float(preds.eq(targets).sum().item())/outs.size(0)

def freeze_batch_norm(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.eval()

def add_args(args1, args2):
    for k, v in args2.__dict__.iteritems():
        args1.__dict__[k] = v
