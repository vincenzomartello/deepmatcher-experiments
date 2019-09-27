import math 
import torch
from torch.nn.functional import softmax
from torch.autograd import Variable

grads = {}


def _save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def crossentropy_gradient(softmax_output,true_labels):
    return (softmax_output-true_labels).data


def get_probabilites(vec):
    probabilities = softmax(vec,dim=0)
    return probabilities


def euclidean_distance_with_max_difference_dimension(v,q):
    distance = 0
    max_difference = -1
    max_difference_dimension = 0
    for idx,vi in enumerate(v):
        qi = q[idx]
        diff = qi-vi
        if(diff>max_difference):
            max_difference = diff
            max_difference_dimension = idx
        distance += (diff)**2
    return math.sqrt(distance),max_difference_dimension


#for each positive sample calculate closer negative sample and its index
def calculate_closer_vector(pos_vector_list,neg_vector_list):
    #mi salvo l'indice del vettore più vicino come chiave
    closer_vectors = []
    for curr_pos_batch in pos_vector_list:
        for curr_positive in curr_pos_batch:
            print('proccessing vector')
            current_min = 100000000
            index = 1
            closer_index = -1
            max_dim = 0
            for batch in neg_vector_list:
                for curr_negative in batch:
                    curr_distance,dim_max = euclidean_distance_with_max_difference_dimension
                    (curr_positive.data,curr_negative.data)
                    if(curr_distance<current_min):
                        current_min=curr_distance
                        closer_index = index
                        max_dim = dim_max
                    index +=1
            closer_vectors.append((closer_index, current_min,dim_max))
    return closer_vectors


def _gradient(output,vector_index,attribute,gradient_loss):
    output[vector_index].backward(gradient_loss,retain_graph=True)
    gradient = grads['classifier'][vector_index]
    start_index = attribute *150
    end_index = start_index+150
    partial_derivative = Variable(torch.cuda.FloatTensor(1200).fill_(0))
    partial_derivative[start_index:end_index] = gradient[start_index:end_index]
    return partial_derivative


#calculate for a sample the minimum vector ri to flip his prediction
def find_smallest_variation_to_change(layer,input_matrix, vector_index,attribute,class_to_reach):
    input_matrix_copy = input_matrix.clone()
    input_matrix_copy.register_hook(_save_grad('classifier'))
    if class_to_reach == 1:
        true_labels = Variable(torch.cuda.FloatTensor([1,0]))
    else:
        true_labels = Variable(torch.cuda.FloatTensor([0,1]))
    x0= input_matrix_copy[vector_index]
    xi = x0
    sum_ri = Variable(torch.cuda.FloatTensor(1200).fill_(0))
    iteration = 0
    
    while(round(get_probabilites(layer.forward(input_matrix_copy)[vector_index])[1].data[0])!=class_to_reach
          and iteration<20):
        output = layer.forward(input_matrix_copy)
        probabilities = get_probabilites(output[vector_index])
        current_match_score = probabilities[1]
        #gradient_loss = crossentropy_gradient(probabilities,true_labels)
        if class_to_reach == 1:
            gradient_loss = torch.cuda.FloatTensor([0,1])
        else:
            gradient_loss = torch.cuda.FloatTensor([1,0])
        current_gradient = _gradient(output,vector_index,attribute,gradient_loss)
        current_norm = torch.norm(current_gradient)
        if (current_norm.data[0]<0.001):
            #sum_ri = Variable(torch.cuda.FloatTensor(1200).fill_(0))
            print("Moving in wrong direction")
            break
        ri = (current_match_score/(current_norm**2)) * current_gradient
        xi = xi+ri
        input_matrix_copy[vector_index].data = input_matrix_copy[vector_index].data.copy_(xi.data)
        sum_ri += ri
        iteration+=1
    if iteration>=20:
        print("can't converge ")
        
    return iteration,sum_ri