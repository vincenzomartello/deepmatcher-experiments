import deepmatcher as dm
from utilities_functions.intermediate_layer_extraction import Hook,return_layer_input_output
import torch
from deepmatcher.data import MatchingIterator

torch.cuda.is_available()

hybrid_model = dm.MatchingModel(attr_summarizer='hybrid')
hybrid_model.load_state('models/hybrid_model.pth')
hybrid_model.cuda()

negative_dataset = dm.data.process(path='sample_data/itunes-amazon/',train='train.csv',
                            validation='validation.csv',test='all_negatives_numericencoding.csv',cache='exp6_neg.pth')

classifier = hybrid_model.classifier
hookF_classifier = []
hookF_classifier.append(Hook(classifier))

batch_size = 32
splits = MatchingIterator.splits(negative_dataset,batch_size=batch_size)

negative_batches = []
for batch in splits[2]:
    negative_batches.append(batch)
    
negative_classifier_inputs = []
negative_classifier_outputs = []
for batch in negative_batches:
    classifier_input,classifier_output = return_layer_input_output(hookF_classifier,batch,hybrid_model)
    negative_classifier_inputs.append(classifier_input)
    negative_classifier_outputs.append(classifier_output)