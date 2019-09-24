from nets.lenet_caffe import LeNetCaffe
from modules.sbp import SBP
from utils.data import get_MNIST
import torch.optim as optim

def load(args):
    net = LeNetCaffe()
    net.build_gate(SBP, [{'kl_scale':40}, {'kl_scale':16}, {}, {}])
    train_loader, test_loader = get_MNIST(100)

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

    args.pretrain_dir = '../results/lenet_caffe_mnist'
    gamma = 1./60000

    return net, gamma, train_loader, test_loader, optimizer, scheduler
