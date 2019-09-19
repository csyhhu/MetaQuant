"""
Code for training baseline quantized model using STE as backward method
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import time

from utils.dataset import get_dataloader
from utils.miscellaneous import accuracy
from utils.quantize import test
from utils.recorder import Recorder

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

################
# Import Model #
################
from models_CIFAR.quantized_resnet import resnet20_cifar, resnet20_stl, resnet56_cifar
# from models_ImageNet.quantized_resnet import resnet18, resnet34, resnet50

# ---------------------------- Configuration --------------------------
import argparse
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Approximation Training')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Optimizer Method')
parser.add_argument('--quantized', '-q', type=str, default='dorefa', help='Quantization Method')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--quantized_head_tail', '-qht', type=boolean_string, default=True,
                    help='Whether to quantize head and tail')
parser.add_argument('--bitW', '-bitW', type=int, default=1, help='Compression ratio')
parser.add_argument('--lr_adjust', '-ad', type=str, default='30', help='LR adjusting method')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
args = parser.parse_args()
# --------------------------------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
dataset_name = args.dataset
n_epoch = args.n_epoch
bitW = args.bitW
quantized_type = args.quantized
optimizer_type = args.optimizer
batch_size = args.batch_size
quantized_head_tail = args.quantized_head_tail
save_root = './Results/%s-%s' % (model_name, dataset_name)
# -------------------------------------------------------------------------

################
# Load Dataset #
################
train_loader = get_dataloader(dataset_name, 'train', batch_size)
test_loader = get_dataloader(dataset_name, 'test', 100)

###################
# Initial Network #
###################
if model_name == 'ResNet20':
    net = resnet20_cifar(bitW=bitW)
# elif model_name == 'ResNet18':
#     net = resnet18(bitW=bitW, quantized_head_tail=quantized_head_tail)
# elif model_name == 'ResNet34':
#     net = resnet34(bitW=bitW, quantized_head_tail=quantized_head_tail)
# elif model_name == 'ResNet50':
#     net = resnet50(bitW=bitW, quantized_head_tail=quantized_head_tail)
else:
    raise NotImplementedError

pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
net.load_state_dict(torch.load(pretrain_path))

if use_cuda:
    print('Dispatch model in %d GPUs' % (torch.cuda.device_count()))
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if optimizer_type == 'SGD-M':
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
elif optimizer_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr)
elif optimizer_type in ['adam', 'Adam']:
    optimizer = optim.Adam(net.parameters(), lr=args.init_lr)
else:
    # optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    raise NotImplementedError

####################
# Initial Recorder #
####################
SummaryPath = '%s/runs-Quant/Baseline-%s-optimizer-%s-%dbits-lr-%s' \
              %(save_root, quantized_type, optimizer_type, bitW, args.lr_adjust)
if args.exp_spec is not '':
    SummaryPath += ('-' + args.exp_spec)

print('Save to %s' %SummaryPath)

if os.path.exists(SummaryPath):
    print('Record exist, remove')
    input()
    shutil.rmtree(SummaryPath)
    os.makedirs(SummaryPath)
else:
    os.makedirs(SummaryPath)

recorder = Recorder(SummaryPath=SummaryPath, dataset_name=dataset_name)

for epoch in range(n_epoch):

    if recorder.stop:
        break

    print('\nEpoch: %d, lr: %e' %(epoch, optimizer.param_groups[0]['lr']))

    net.train()
    end = time.time()
    recorder.reset_performance()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Gradient descent
        optimizer.zero_grad()
        outputs = net(inputs, quantized_type)
        losses = nn.CrossEntropyLoss()(outputs, targets)
        losses.backward()
        optimizer.step()

        recorder.update(loss=losses.item(), acc=accuracy(outputs.data, targets.data, (1, 5)),
                        batch_size=outputs.shape[0], cur_lr=optimizer.param_groups[0]['lr'], end=end)

        recorder.print_training_result(batch_idx, len(train_loader))
        end = time.time()

    test_acc = test(net=net, quantized_type=quantized_type,
                    test_loader=test_loader, dataset_name=dataset_name, n_batches_used=100)
    recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)

    # Adjust lr
    recorder.adjust_lr(optimizer=optimizer, adjust_type=args.lr_adjust, epoch=epoch)

best_test_acc = recorder.get_best_test_acc()
if type(best_test_acc) == tuple:
    print('Best test top 1 acc: %.3f, top 5 acc: %.3f' % (best_test_acc[0], best_test_acc[1]))
else:
    print('Best test acc: %.3f' %best_test_acc)
recorder.close()