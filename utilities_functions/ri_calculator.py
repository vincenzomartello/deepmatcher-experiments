import math 
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


grads = {}


def _save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook



def get_probabilites(vec):
    probabilities = F.softmax(vec,dim=0)
    return probabilities


#calculate for a sample the minimum vector ri to flip his prediction
def find_smallest_variation_to_change_v2(layer,classifier_length,attribute_length,input_matrix, 
                                      vector_index,attributes,class_to_reach,lr0,decay):
    max_iterations = 1000
    input_matrix_copy = input_matrix.clone()
    input_matrix_copy.register_hook(_save_grad('classifier'))
    if class_to_reach == 1:
        desidered_labels = Variable(torch.cuda.LongTensor([1]*len(input_matrix)))
    else:
        desidered_labels = Variable(torch.cuda.LongTensor([0]*len(input_matrix)))
    x0= input_matrix_copy[vector_index]
    xi = x0
    sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
    iterations = 1
    continue_search = True
    while(continue_search and iterations <1000 and round(get_probabilites(layer.forward(input_matrix_copy)[vector_index])[1].data[0])!=class_to_reach):      
        output = layer.forward(input_matrix_copy)
        probabilities = get_probabilites(output[vector_index])
        ##f(x) is the probability of the current state
        if class_to_reach == 1:
            fx = 1 - probabilities[1]
        else:
            fx = probabilities[1]
        #- to move to the opposite direction of gradients
        loss = F.cross_entropy(F.softmax(output,dim=1),desidered_labels)
        loss.backward()
        current_gradient = grads['classifier'][vector_index]
        partial_derivative = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
        for att in attributes:
            start_index = att * attribute_length
            end_index = start_index+ attribute_length
            partial_derivative[start_index:end_index] = current_gradient[start_index:end_index]
        current_norm = torch.norm(partial_derivative)
        if current_norm.data[0] <=0.00001:
            sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
            print(" Gradient is null")
            continue_search = False
        else:
            lr = lr0 * exp(-decay*iterations)
            ri = lr *(-partial_derivative)
            xi = xi+ri
            input_matrix_copy[vector_index].data = input_matrix_copy[vector_index].data.copy_(xi.data)
            sum_ri += ri
            iterations +=1
    if iterations>=max_iterations:
        sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
        print("can't converge in {} iterations".format(str(iterations)))
        
    return sum_ri


def find_smallest_variation_to_change_v1(layer,classifier_length,attribute_length,input_matrix, 
                                      vector_index,attributes,class_to_reach):
    max_iterations = 1000
    input_matrix_copy = input_matrix.clone()
    input_matrix_copy.register_hook(_save_grad('classifier'))
    if class_to_reach == 1:
        desidered_labels = Variable(torch.cuda.FloatTensor([[0,1]]*len(input_matrix)))
    else:
        desidered_labels = Variable(torch.cuda.FloatTensor([[1,0]]*len(input_matrix)))
    x0= input_matrix_copy[vector_index]
    xi = x0
    sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
    iterations = 1
    continue_search = True
    while(continue_search and iterations <1000 and round(get_probabilites(layer.forward(input_matrix_copy)[vector_index])[1].data[0])!=class_to_reach):      
        output = layer.forward(input_matrix_copy)
        probabilities = get_probabilites(output[vector_index])
        ##f(x) is the probability of the current state
        if class_to_reach == 1:
            fx = 1 - probabilities[1]
        else:
            fx = probabilities[1]
        #- to move to the opposite direction of gradients
        loss = F.binary_cross_entropy(F.softmax(output,dim=1),desidered_labels)
        loss.backward()
        current_gradient = grads['classifier'][vector_index]
        partial_derivative = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
        for att in attributes:
            start_index = att * attribute_length
            end_index = start_index+ attribute_length
            partial_derivative[start_index:end_index] = current_gradient[start_index:end_index]
        current_norm = torch.norm(partial_derivative)
        if current_norm.data[0] <=0.00001:
            sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
            print(" Gradient is null")
            continue_search = False
        else:
            ri = -(fx/current_norm)*(partial_derivative)
            xi = xi+ri
            input_matrix_copy[vector_index].data = input_matrix_copy[vector_index].data.copy_(xi.data)
            sum_ri += ri
            iterations +=1
    if iterations>=max_iterations:
        sum_ri = Variable(torch.cuda.FloatTensor(classifier_length).fill_(0))
        print("can't converge in {} iterations".format(str(iterations)))
        
    return sum_ri