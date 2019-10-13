import math 
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable


grads = {}


def _save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def crossentropy_gradient(softmax_output,true_labels,class_to_reach):
    return (softmax_output-true_labels).data
 


def get_probabilites(vec):
    probabilities = softmax(vec,dim=0)
    return probabilities


def _gradient(output,classifier_length,attribute_length,vector_index,attributes,gradient_loss):
    output[vector_index].backward(gradient_loss,retain_graph=True)
    gradient = grads['classifier'][vector_index]
    partial_derivative = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
    for att in attributes:
        start_index = att * attribute_length
        end_index = start_index+ attribute_length
        partial_derivative[start_index:end_index] = gradient[start_index:end_index]
    return partial_derivative


#calculate for a sample the minimum vector ri to flip his prediction
def find_smallest_variation_to_change(layer,classifier_length,attribute_length,input_matrix, 
                                      vector_index,attributes,class_to_reach):
    input_matrix_copy = input_matrix.clone()
    input_matrix_copy.register_hook(_save_grad('classifier'))
    if class_to_reach == 1:
        true_labels = Variable(torch.cuda.FloatTensor([0,1]))
    else:
        true_labels = Variable(torch.cuda.FloatTensor([1,0]))
    x0= input_matrix_copy[vector_index]
    xi = x0
    sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
    iteration = 0
    
    while(round(get_probabilites(layer.forward(input_matrix_copy)[vector_index])[1].data[0])!=class_to_reach
          and iteration<50):
        output = layer.forward(input_matrix_copy)
        probabilities = get_probabilites(output[vector_index])
        current_match_score = probabilities[1]
        ##f(x) is the probability of the current state
        if class_to_reach == 1:
            fx = 1 - current_match_score
            gradient_loss = torch.cuda.FloatTensor([0,1])
        else:
            fx = current_match_score
            gradient_loss = torch.cuda.FloatTensor([1,0])
    
        ##gradient_loss = crossentropy_gradient(probabilities,true_labels,class_to_reach)
        
        current_gradient = _gradient(output,classifier_length,attribute_length,vector_index,attributes,gradient_loss)
        current_norm = torch.norm(current_gradient)
        if (current_norm.data[0]<0.001):
            sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
            print("Moving in wrong direction")
            break
        ri = (fx/(current_norm)) * current_gradient
        xi = xi+ri
        input_matrix_copy[vector_index].data = input_matrix_copy[vector_index].data.copy_(xi.data)
        sum_ri += ri
        iteration+=1
    if iteration>=50:
        sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
        print("can't converge ")
        
    return iteration,sum_ri