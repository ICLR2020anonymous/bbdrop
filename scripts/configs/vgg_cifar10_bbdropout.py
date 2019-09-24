from nets.vgg import VGG
from modules.bbdropout import BBDropout
from utils.data import get_CIFAR10
from utils.misc import add_args
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_dir', default='../results/vgg_cifar10', type=str)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--gamma', default=1./60000, type=float)
parser.add_argument('--kl_scale', default=1.0, type=float)
sub_args, _ = parser.parse_known_args()

def load(args):
    add_args(args, sub_args)
    net = VGG(10)
    net.build_gate(BBDropout, {'a_uc_init':2.0, 'kl_scale':args.kl_scale})
    train_loader, test_loader = get_CIFAR10(args.batch_size)

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
