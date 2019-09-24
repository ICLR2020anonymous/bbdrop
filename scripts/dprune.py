import argparse
import torch
import torch.nn as nn
import logging
import os
import sys
import imp
import numpy as np
sys.path.append('../')
from utils.accumulator import Accumulator
from utils.misc import accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/lenet_mlp_mnist_dbbdropout.py', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--save_freq', default=20, type=int)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--run_name', default='trial', type=str)
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.backends.cudnn.benchmark = True
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# load models from config file
config = os.path.splitext(os.path.basename(args.config))[0]
net, train_loader, test_loader, optimizer, scheduler = \
        imp.load_source(config, args.config).load(args)
net.cuda()
cent_fn = nn.CrossEntropyLoss().cuda()
save_dir = os.path.join('../results', config, args.run_name)
accm = Accumulator('cent', 'acc')

def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for v in vars(args):
            f.write('{}: {}\n'.format(v, getattr(args, v)))

    ckpt = torch.load(os.path.join(args.pretrain_dir, 'model.tar'))
    net.load_state_dict(ckpt['state_dict'], strict=False)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(config)
    logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w'))
    logger.info(str(args) + '\n')

    for epoch in range(1, args.num_epochs+1):
        accm.reset()
        scheduler.step()
        line = 'epoch {} starts with lr'.format(epoch)
        for pg in optimizer.param_groups:
            line += ' {:.3e}'.format(pg['lr'])
        logger.info(line)
        net.train()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            outs = net(x)
            cent = cent_fn(outs, y)
            reg = net.get_reg_dep().cuda()
            loss = cent + args.gamma*reg
            loss.backward()
            optimizer.step()
            accm.update([cent.item(), accuracy(outs, y)])
        line = accm.info(header='train', epoch=epoch)

        if epoch % args.eval_freq == 0:
            logger.info(line)
            test(load=False, logger=logger, epoch=epoch)
        else:
            logger.info(line + '\n')

        if epoch % args.save_freq == 0:
            torch.save({'state_dict':net.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    test(load=False)
    torch.save({'state_dict':net.state_dict()},
            os.path.join(save_dir, 'model.tar'))

def test(load=True, logger=None, epoch=None):
    if load:
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])

    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(config)
        logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'test.log'), mode='w'))

    net.eval()
    net.reset_dep()
    accm.reset()
    for it, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        outs = net(x)
        cent = cent_fn(outs, y)
        accm.update([cent.item(), accuracy(outs, y)])
    logger.info(accm.info(header='test', epoch=epoch))
    logger.info('reg {:.4f}'.format(net.get_reg_dep().item()))
    logger.info('pruned size {}'.format(str(net.get_pruned_size())))
    logger.info('pruned size (dep) {}'.format(str(net.get_pruned_size_dep())))
    logger.info('speedup in flops {:.4f}'.format(net.get_speedup_dep()))
    logger.info('memory saving {:.4f}\n'.format(net.get_memory_saving_dep()))

if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()
