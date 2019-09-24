from nets.vgg import VGG
from modules.l0reg import L0Reg
from utils.data import get_CIFAR10
from utils.misc import add_args
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_dir', default='../results/vgg_cifar10', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--lamb', default=1e-7, type=float)
sub_args, _ = parser.parse_known_args()

def load(args):
    add_args(args, sub_args)
    net = VGG(10)
    net.build_gate(L0Reg, {'weight_decay':1e-4, 'lamb':args.lamb, 'droprate_init':0.2})
    train_loader, test_loader = get_CIFAR10(args.batch_size)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(r*args.num_epochs) for r in [.5, .8]],
            gamma=0.1)

    return net, train_loader, test_loader, optimizer, scheduler
