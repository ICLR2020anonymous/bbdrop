from nets.vgg import VGG
from modules.vib import VIB
from utils.data import get_CIFAR100
from utils.data import add_args
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_dir', default='../results/vgg_cifar100', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--gamma', default=1e-6, type=float)
sub_args, _ = parser.parse_known_args()

def load(args):
    add_args(args, sub_args)
    net = VGG(100)
    net.build_gate(VIB)
    train_loader, test_loader = get_CIFAR100(args.batch_size)

    base_params = []
    gate_params = []
    for name, param in net.named_parameters():
        if 'gate' in name:
            gate_params.append(param)
        else:
            base_params.append(param)
    optimizer = optim.Adam([
        {'params':gate_params, 'lr':1e-2},
        {'params':base_params, 'lr':1e-3, 'weight_decay':1e-4}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(r*args.num_epochs) for r in [.5, .8]],
            gamma=0.1)

    return net, train_loader, test_loader, optimizer, scheduler
