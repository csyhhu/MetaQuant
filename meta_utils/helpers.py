"""
Some helper function for meta inference
"""

import torch
from utils.miscellaneous import get_layer
import numpy as np


def meta_gradient_generation(meta_net, net, meta_method, meta_hidden_state_dict=None, fix_meta=False):

    meta_grad_dict = dict()
    new_meta_hidden_state_dict = dict()
    layer_name_list = net.layer_name_list

    for idx, layer_info in enumerate(layer_name_list):

        layer_name = layer_info[0]
        layer_idx = layer_info[1]

        layer = get_layer(net, layer_idx)

        grad = layer.quantized_grads.data
        pre_quantized_weight = layer.pre_quantized_weight.data
        bias = layer.bias

        if bias is not None:
            bias_grad = bias.grad.data.clone()
        else:
            bias_grad = None

        if meta_method == 'FC-Grad':
            meta_input = grad.data.view(-1, 1)

            if fix_meta:
                with torch.no_grad():
                    meta_grad = meta_net(meta_input)
            else:
                meta_grad = meta_net(meta_input)

        elif meta_method == 'simple':
            meta_grad = grad.data

        elif meta_method in ['MultiFC']:

            flatten_grad = grad.data.view(-1, 1)
            flatten_weight = pre_quantized_weight.data.view(-1, 1)

            if fix_meta:
                with torch.no_grad():
                    meta_output = meta_net(flatten_weight)
            else:
                meta_output = meta_net(flatten_weight)

            meta_grad = flatten_grad * meta_output

        elif meta_method in ['LSTMFC']:

            flatten_grad = grad.data.view(1, -1, 1)
            flatten_weight = pre_quantized_weight.data.view(1, -1, 1)

            if meta_hidden_state_dict is not None and layer_name in meta_hidden_state_dict:
                meta_hidden_state = meta_hidden_state_dict[layer_name]
            else:
                meta_hidden_state = None

            if fix_meta:
                with torch.no_grad():
                    meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)
            else:
                meta_output, hidden = meta_net(flatten_weight, meta_hidden_state)

            meta_grad = flatten_grad * meta_output

        else:
            raise NotImplementedError

        # Reshape the flattened meta gradient into the original shape
        meta_grad = meta_grad.reshape(grad.shape)

        if bias is not None:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, bias_grad.data)
        else:
            meta_grad_dict[layer_name] = (layer_idx, meta_grad, None)

        # Assigned pre_quantized_grads with meta grad for weights update
        layer.pre_quantized_grads = meta_grad.data.clone()

    return meta_grad_dict, new_meta_hidden_state_dict


def update_parameters(net, lr):
    for param in net.parameters():
        # if torch.sum(torch.abs(param.grad.data)) == 0:
        #     print('[Warning] Gradient is 0, missing assigned?')
        param.data.add_(-lr * param.grad.data)