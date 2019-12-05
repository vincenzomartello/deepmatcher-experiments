#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import deepmatcher as dm
from deepmatcher.data import MatchingIterator
from torch.autograd import Variable

# class to save intermediate layer output/input

class Hook:

    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(
        self,
        module,
        input,
        output,
        ):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

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


def return_layer_input_output(dataset_dir,dataset_name,batch_size,model,layer):
    dataset = dm.data.process(path = dataset_dir,train= dataset_name+'.csv',left_prefix='ltable_',right_prefix='rtable_',cache=dataset_name+'.pth')
    hook = Hook(layer)
    dataset_tuple = dataset,
    splits = MatchingIterator.splits(dataset_tuple,batch_size=batch_size)
    tupleids = []
    layer_inputs = []
    layer_outputs = []
    for batch in splits[0]:
        tupleids.append(batch.id)
        layer_inp,layer_out = return_layer_input_output_for_batch(model,hook,batch)
        layer_inputs.append(layer_inp[0])
        layer_outputs.append(layer_out[0])
    return layer_inputs,layer_outputs,list(map(int,_flat_list(tupleids)))


def _return_input(module,module_input,module_output):
    global current_layer_input
    current_layer_input = Variable(module_input[0].data.cuda(),requires_grad=True)


def return_layer_input(dataset_dir,dataset_name,batch_size,model,layer,device = 0):
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
    return layer_inputs,list(map(int,_flat_list(tupleids)))



