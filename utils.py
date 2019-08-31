import torch

# class to save intermediate layer output/input
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def return_layer_input_output(hook_functions,batch,model):
    out= model(batch)
    layer_inputs = []
    layer_outputs = []
    for hook in hook_functions:
        layer_inputs.append(hook.input)
        layer_outputs.append(hook.output)
    return layer_inputs,layer_outputs