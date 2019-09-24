from nets.lenet_caffe import LeNetCaffe
from utils.data import get_MNIST
from utils.misc import add_args
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int)
sub_args, _ = parser.parse_known_args()

def load(args):
    add_args(args, sub_args)
    net = LeNetCaffe()
    train_loader, test_loader = get_MNIST(args.batch_size)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(r*args.num_epochs) for r in  [0.5, 0.8]],
            gamma=0.1)
    return net, train_loader, test_loader, optimizer, scheduler
