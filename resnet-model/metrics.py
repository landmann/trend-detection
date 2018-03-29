import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import models

from train_set import MiniPlacesDataset
from fine_tuning_config_file import *

import sys

import os

# Outputs a file containing top5 outputs and actual label.
def compute_metrics(model, use_gpu, output_path, test_options):
    print(output_path)
    mode = output_path.split('/')[-1]
    mode = mode.split('_')[0]
    data_dir = os.path.expanduser(DATA_PATH)
    test_options['labels_path'] = os.path.join(data_dir, mode + '.txt')
    print(test_options)

    testloader = MiniPlacesDataset(
        photos_path = test_options['photos_path'],
        labels_path = test_options['labels_path'],
        transform = test_options['transform']
    ) 
    data_loader = torch.utils.data.DataLoader(testloader, batch_size=1) 
    model.eval()
    output_labels = []
    for data in data_loader: 
        inputs, labels = data

        inputs = Variable(inputs.float().cuda())
        labels = Variable(labels.long().cuda())    
         
        output = model(inputs)
        probabilities, prediction = output.topk(5, dim=1, largest=True, sorted=True)

        prediction = [str(elt) for elt in prediction.data[0]]
        prediction += [str(prb) for prb in probabilities.data[0]]
        prediction.append(str(labels.data[0]))
        prediction = " ".join(prediction)
        print(prediction)
        output_labels.append(prediction)
    
    output_labels = "\n".join(output_labels)
    with open(output_path, 'w') as outfile:
        outfile.write(output_labels)
                  


