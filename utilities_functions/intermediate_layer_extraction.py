#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import deepmatcher as dm
from deepmatcher.data import MatchingIterator

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

def return_layer_input_output(dataset_dir,dataset_name,batch_size,model,layer):
    dataset = dm.data.process(path = dataset_dir,train= dataset_name+'.csv',
                            validation='validation.csv',left_prefix='ltable_',right_prefix='rtable_',cache=dataset_name+'.pth')
    hook = Hook(layer)
    batch_size = 32
    splits = MatchingIterator.splits(dataset,batch_size=batch_size)
    batches = []
    tupleids = []
    for batch in splits[0]:
        batches.append(batch)
        tupleids.append(batch.id)
    layer_inputs = []
    layer_outputs = []
    for batch in batches:
        layer_inp,layer_out = return_layer_input_output_for_batch(model,hook,batch)
        layer_inputs.append(layer_inp)
        layer_outputs.append(layer_out)
    return layer_inputs,layer_outputs,list(map(int,_flat_list(tupleids)))
    
