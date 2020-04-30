"""
A simplification version of meta-quantize for multiple experiments
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utils.dataset import get_dataloader
from meta_utils.meta_network import MetaFC, MetaLSTMFC, MetaDesignedMultiFC
from meta_utils.SGD import SGD
from meta_utils.adam import Adam
from meta_utils.helpers import meta_gradient_generation, update_parameters
from utils.recorder import Recorder
from utils.miscellaneous import AverageMeter, accuracy, progress_bar
from utils.miscellaneous import get_layer
from utils.quantize import test

##################
# Import Network #
##################
from models_CIFAR.quantized_meta_resnet import resnet20_cifar, resnet20_stl, resnet56_cifar, resnet32_cifar
# from models_ImageNet.quantized_meta_resnet import resnet18, resnet34, resnet50

import argparse
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Meta Quantization')
parser.add_argument('--model', '-m', type=str, default='ResNet20', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Optimizer Method')
parser.add_argument('--quantize', '-q', type=str, default='dorefa', help='Quantization Method')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--bitW', '-bw', type=int, default=1, help='Quantization Bit')
parser.add_argument('--meta_type', '-meta', type=str, default='MultiFC', help='Type of Meta Network')
parser.add_argument('--hidden_size', '-hidden', type=int, default=100,
                    help='Hidden size of meta network')
parser.add_argument('--num_fc', '-nfc', type=int, default=3,
                    help='Number of layer of FC in MultiFC')
parser.add_argument('--num_lstm', '-nlstm', type=int, default=2,
                    help='Number of layer of LSTM in MultiLSTMFC')
parser.add_argument('--n_epoch', '-n', type=int, default=100,
                    help='Maximum training epochs')
parser.add_argument('--fix_meta', '-fix', type=boolean_string, default='False',
                    help='Whether to fix meta')
parser.add_argument('--fix_meta_epoch', '-n_fix', type=int, default=0,
                    help='When to fix meta')
parser.add_argument('--random', '-r', type=str, default=None,
                    help='Whether to use random layer')
parser.add_argument('--meta_nonlinear', '-nonlinear', type=str, default=None,
                    help='Nonlinear used in meta network')
parser.add_argument('--lr_adjust', '-ad', type=str,
                    default='30', help='LR adjusting method')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
parser.add_argument('--weight_decay', '-decay', type=float, default=0,
                    help='Weight decay for training meta quantizer')
args = parser.parse_args()

# ------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
dataset_name = args.dataset
meta_method = args.meta_type # ['LSTM', 'FC', 'simple']
MAX_EPOCH = args.n_epoch
optimizer_type = args.optimizer # ['SGD', 'SGD-M', 'adam']
hidden_size = args.hidden_size
num_lstm = args.num_lstm
num_fc = args.num_fc
random_type = args.random
lr_adjust = args.lr_adjust
batch_size = args.batch_size
bitW = args.bitW
quantized_type = args.quantize
save_root = './Results/%s-%s' % (model_name, dataset_name)
# ------------------------------------------
print(args)
# input('Take a look')

import utils.global_var as gVar
gVar.meta_count = 0

###################
# Initial Network #
###################
if model_name == 'ResNet20':
    net = resnet20_cifar(bitW=bitW)
# elif model_name == 'ResNet18':
#     net = resnet18(bitW=bitW, quantized_head_tail=args.quantized_head_tail)
# elif model_name == 'ResNet34':
#     net = resnet34(bitW=bitW, quantized_head_tail=args.quantized_head_tail)
# elif model_name == 'ResNet50':
#     net = resnet50(bitW=bitW, quantized_head_tail=args.quantized_head_tail)
else:
    raise NotImplementedError

pretrain_path = '%s/%s-%s-pretrain.pth' % (save_root, model_name, dataset_name)
net.load_state_dict(torch.load(pretrain_path), strict=False)

# Get layer name list
layer_name_list = net.layer_name_list
assert (len(layer_name_list) == gVar.meta_count)
print('Layer name list completed.')

if use_cuda:
    net.cuda()

################
# Load Dataset #
################
train_loader = get_dataloader(dataset_name, 'train', batch_size)
test_loader = get_dataloader(dataset_name, 'test', 100)

##########################
# Construct Meta-Network #
##########################
if meta_method == 'LSTMFC':
    meta_net = MetaLSTMFC(hidden_size=hidden_size)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-nlstm-1-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size,
                     quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method in ['FC-Grad']:
    meta_net = MetaFC(hidden_size=hidden_size, use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size,
                     quantized_type, optimizer_type, bitW, lr_adjust)
elif meta_method == 'MultiFC':
    meta_net = MetaDesignedMultiFC(hidden_size=hidden_size,
                                   num_layers = args.num_fc,
                                   use_nonlinear=args.meta_nonlinear)
    SummaryPath = '%s/runs-Quant/Meta-%s-Nonlinear-%s-' \
                  'hidden-size-%d-nfc-%d-%s-%s-%dbits-lr-%s' \
                  % (save_root, meta_method, args.meta_nonlinear, hidden_size, num_fc,
                     quantized_type, optimizer_type, bitW, lr_adjust)
else:
    raise NotImplementedError

print(meta_net)

if use_cuda:
    meta_net.cuda()

meta_optimizer = optim.Adam(meta_net.parameters(), lr=1e-3, weight_decay=args.weight_decay)
meta_hidden_state_dict = dict() # Dictionary to store hidden states for all layers for memory-based meta network
meta_grad_dict = dict() # Dictionary to store meta net output: gradient for origin network's weight / bias

##################
# Begin Training #
##################
# meta_opt_flag = True # When it is false, stop updating meta optimizer

# Optimizer for original network, just for zeroing gradient and get refined gradient
if optimizer_type == 'SGD-M':
    optimizee = SGD(net.parameters(), lr=args.init_lr,
                    momentum=0.9, weight_decay=5e-4)
elif optimizer_type == 'SGD':
    optimizee = SGD(net.parameters(), lr=args.init_lr)
elif optimizer_type in ['adam', 'Adam']:
    optimizee = Adam(net.parameters(), lr=args.init_lr,
                     weight_decay=5e-4)
else:
    raise NotImplementedError

####################
# Initial Recorder #
####################
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

##################
# Begin Training #
##################
meta_grad_dict = dict()
for epoch in range(MAX_EPOCH):

    if recorder.stop: break

    print('\nEpoch: %d, lr: %e' % (epoch, optimizee.param_groups[0]['lr']))

    net.train()
    end = time.time()

    recorder.reset_performance()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        meta_optimizer.zero_grad()

        # Construct new net
        if batch_idx == 0 and epoch == 0:
            pass
            # meta_grad_dict, meta_hidden_state_dict = construct_init_meta_grad_dict(net)
        else:
            meta_grad_dict, meta_hidden_state_dict = \
                meta_gradient_generation(
                        meta_net, net, meta_method, meta_hidden_state_dict
                )

        # Assign meta gradient for actual gradients used in update_parameters
        if len(meta_grad_dict) != 0:
            for layer_info in net.layer_name_list:
                layer_name = layer_info[0]
                layer_idx = layer_info[1]
                layer = get_layer(net, layer_idx)
                layer.weight.grad.data = (layer.calibration * layer.pre_quantized_grads)
                # layer.weight.grad.data.copy_(layer.calibration * meta_grad_dict[layer_name][1].data)

        # Get refine gradients for next computation
        optimizee.get_refine_gradient()

        outputs = net(inputs, quantized_type=quantized_type,
                      meta_grad_dict=meta_grad_dict,
                      lr=optimizee.param_groups[0]['lr'])

        optimizee.zero_grad()

        # Taking backward generate gradient for meta pruner and base model
        # Non-meta-weights' (bias, BN layer) gradient is attained here
        losses = nn.CrossEntropyLoss()(outputs, targets)

        meta_optimizer.step()

        # These gradient should be saved in next iteration's inference
        if len(meta_grad_dict) != 0:
            update_parameters(net, lr=optimizee.param_groups[0]['lr'])

        losses.backward()

        recorder.update(loss=losses.data.item(), acc=accuracy(outputs.data, targets.data, (1,5)),
                        batch_size=outputs.shape[0], cur_lr=optimizee.param_groups[0]['lr'], end=end)

        recorder.print_training_result(batch_idx, len(train_loader))
        end = time.time()

    test_acc = test(net, quantized_type=quantized_type, test_loader=test_loader,
                    dataset_name=dataset_name, n_batches_used=None)
    recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)

    # Adjust learning rate
    recorder.adjust_lr(optimizer=optimizee, adjust_type=lr_adjust, epoch=epoch)

best_test_acc = recorder.get_best_test_acc()
if type(best_test_acc) == tuple:
    print('Best test top 1 acc: %.3f, top 5 acc: %.3f' % (best_test_acc[0], best_test_acc[1]))
else:
    print('Best test acc: %.3f' %best_test_acc)
recorder.close()