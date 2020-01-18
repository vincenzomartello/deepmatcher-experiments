#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import deepmatcher as dm
from deepmatcher.data import MatchingIterator
import torch

# methods to save intermediate layer output/input

def _flat_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def return_layer_input_output_for_batch(model,hook,batch):
    out = model(batch)
    return (hook.input,hook.output)


def _return_layer_input_for_batch(model,hook,batch):
    out = model(batch)
    return hook.input


def _return_input(module,module_input,module_output):
    global current_layer_input
    if isinstance(module_input,tuple):
        current_layer_input = (module_input[0].detach().requires_grad_())
    else:
        current_layer_input = (module_input[0].detach().requires_grad_())


def _return_input_output(module,module_input,module_output):
    global current_layer_input
    global current_layer_output
    if isinstance(module_input,tuple):
        current_layer_input = (module_input[0].detach().requires_grad_())
    else:
        current_layer_input = (module_input[0].detach().requires_grad_())
    if isinstance(module_output,tuple):
        current_layer_output = (module_output[0].detach().requires_grad_())
    else:
        current_layer_output = (module_output.detach().requires_grad_())


def return_layer_input_output(dataset_dir,dataset_name,batch_size,model,layer,device='cuda'):
    dataset = dm.data.process(path=dataset_dir,train=dataset_name+'.csv',left_prefix='ltable_',right_prefix='rtable_',cache=dataset_name+'.pth')
    dataset_tuple = dataset,
    splits = MatchingIterator.splits(dataset_tuple,batch_size=batch_size, device = 'cuda')
    tupleids = []
    layer_inputs = []
    layer_outputs = []
    hook = layer.register_forward_hook(_return_input_output)
    for batch in splits[0]:
        tupleids.append(batch.id)
        model.forward(batch)
        layer_inputs.append(current_layer_input)
        layer_outputs.append(current_layer_output)
    hook.remove()
    id_flattened = list(map(int,_flat_list(tupleids)))
    ##map in which for each sample id we have the corrisponde vector in intermediate layer
    res = {}
    j = 0
    for batch1,batch2 in zip(layer_inputs,layer_outputs):
        for inp,out in zip(batch1,batch2):
            res[id_flattened[j]] = (inp,out)
            j += 1
    return res


def return_layer_input(model,layer,dataset_dir,dataset_name,batch_size=32,device ='cuda'):
    dataset = dm.data.process(path=dataset_dir,train=dataset_name+'.csv',left_prefix='ltable_',right_prefix='rtable_',cache=dataset_name+'.pth')
    dataset_tuple = dataset,
    splits = MatchingIterator.splits(dataset_tuple,batch_size=batch_size, device = device)
    tupleids = []
    layer_inputs = []
    hook = layer.register_forward_hook(_return_input)
    for batch in splits[0]:
        tupleids.append(batch.id)
        model.forward(batch)
        layer_inputs.append(current_layer_input)
    hook.remove()
    id_flattened = _flat_list(tupleids)
    ##map in which for each sample id we have the corrisponde vector in intermediate layer
    res = {}
    j = 0
    for batch in layer_inputs:
        for sample in batch:
            res[id_flattened[j]] = sample
            j += 1
    return res



