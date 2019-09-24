import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--dep', action='store_true')
parser.add_argument('--run_header', default=None, type=str)
parser.add_argument('--num_trials', default=3, type=int)
parser.add_argument('--stat', default='median', type=str)
args = parser.parse_args()
assert(args.config is not None and args.run_header is not None)

acc = [0]*args.num_trials
pruned_size = [0]*args.num_trials
flops = [0]*args.num_trials
memory = [0]*args.num_trials

for i in range(args.num_trials):
    filename = os.path.join('../results', args.config,
            args.run_header + '_{}'.format(i), 'test.log')
    with open(filename, 'r') as f:
        lines = f.readlines()
        if args.dep:
            acc[i] = float(lines[0].split()[4][:-1])
            tokens = lines[3].split()[3:]
            tokens[0] = tokens[0][1:]
            pruned_size[i] = [int(tok[:-1]) for tok in tokens]
            flops[i] = float(lines[4].split()[3])
            memory[i] = float(lines[5].split()[2])
        else:
            acc[i] = float(lines[0].split()[4][:-1])
            tokens = lines[2].split()[2:]
            tokens[0] = tokens[0][1:]
            pruned_size[i] = [int(tok[:-1]) for tok in tokens]
            flops[i] = float(lines[3].split()[3])
            memory[i] = float(lines[4].split()[2])

acc = np.array(acc)
pruned_size = np.array(pruned_size)
flops = np.array(flops)
memory = np.array(memory)

print acc
print flops
print memory

print list(np.median(pruned_size, 0).astype(int))
if args.stat == 'mean':
    print 'acc {:.4f}+-{:.4f}'.format(np.mean(acc), np.std(acc))
    print 'err {:.4f}+-{:.4f}'.format(np.mean(1-acc), np.std(1-acc))
else:
    print 'acc {:.4f}'.format(np.median(acc))
    print 'err {:.4f}+-{:.4f}'.format(np.mean(1-acc), np.std(1-acc))
print 'flops {:.4f}'.format(np.median(flops))
print 'memory {:.4f}'.format(np.median(memory))
