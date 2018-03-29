### Section 1 - First, let's import everything we will be needing.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
from fine_tuning_config_file import *
#import torchsample

# custom datasets
#from train_set_with_flips import GraphTrendDataset
from train_set import GraphTrendDataset
from test_set import GraphTrendTestDataset as TestSet
from runningAvg import RunningAvg
from accuracy import accuracy

import custom_transforms

if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


## If you want to use the GPU, set GPU_MODE TO 1 in config file

use_gpu = GPU_MODE
print('Are you using your GPU? {}'.format("Yes!" if use_gpu else "Nope :("))
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)


### SECTION 2 - data loading and shuffling/augmentation/normalization : all handled by torch automatically.

# This is a little hard to understand initially, so I'll explain in detail here!

# For training, the data gets transformed by undergoing augmentation and normalization. 
# The RandomSizedCrop basically takes a crop of an image at various scales between 0.01 to 0.8 times the size of the image and resizes it to given number
# Horizontal flip is a common technique in computer vision to augment the size of your data set. Firstly, it increases the number of times the network gets
# to see the same thing, and secondly it adds rotational invariance to your networks learning.


# Just normalization for validation, no augmentation. 

# You might be curious where these numbers came from? For the most part, they were used in popular architectures like the AlexNet paper. 
# It is important to normalize your dataset by calculating the mean and standard deviation of your dataset images and making your data unit normed. However,
# it takes a lot of computation to do so, and some papers have shown that it doesn't matter too much if they are slightly off. So, people just use imagenet
# dataset's mean and standard deviation to normalize their dataset approximately. These numbers are imagenet mean and standard deviation!

# If you want to read more, transforms is a function from torchvision, and you can go read more here - http://pytorch.org/docs/master/torchvision/transforms.html

#TODO: CHANGE THE TRANSFORMS
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        custom_transforms.all_transforms,
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        #transforms.Scale((224, 224)),
        transforms.ToTensor()
#        torchsample.transforms.RandomRotate(30),
#        torchsample.transforms.RandomGamma(0.5, 1.5),
#        torchsample.transforms.RandomSaturation(-0.8, 0.8),
#        torchsample.transforms.RandomBrightness(-0.3, 0.3),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}



# Enter the absolute path of the dataset folder below. Keep in mind that this code expects data to be in same format as Imagenet. I encourage you to
# use your own dataset. In that case you need to organize your data such that your dataset folder has EXACTLY two folders. Name these 'train' and 'val'
# Yes, this is case sensitive. The 'train' folder contains training set and 'val' fodler contains validation set on which accuracy is measured. 

# The structure within 'train' and 'val' folders will be the same. They both contain one folder per class. All the images of that class are inside the 
# folder named by class name.

# So basically, if your dataset has 3 classes and you're trying to classify between pictures of 1) dogs 2) cats and 3) humans,
# say you name your dataset folder 'data_directory'. Then inside 'data_directory' will be 'train' and 'test'. Further, Inside 'train' will be 
# 3 folders - 'dogs', 'cats', 'humans'. All training images for dogs will be inside this 'dogs'. Similarly, within 'val' as well there will be the same
# 3 folders. 

## So, the structure looks like this : 
# data_dar
#      |- train 
#            |- dogs
#                 |- dog_image_1
#                 |- dog_image_2
#                        .....

#            |- cats
#                 |- cat_image_1
#                 |- cat_image_1
#                        .....
#            |- humans
#      |- val
#            |- dogs
#            |- cats
#            |- humans

data_dir = os.path.expanduser(DATA_PATH)

dsets = {}
for mode in ['train', 'val', 'test']: 
    kwargs = {
            'photos_path': os.path.join(data_dir, mode + '/'),
            'labels_path': os.path.join(data_dir, mode + '_labels.txt'),
            'transform': data_transforms[mode]
    }
    if mode == 'test': 
        #kwargs['flip'] = True
        dsets[mode] = TestSet(**kwargs)
    else: 
        dsets[mode] = GraphTrendDataset(**kwargs)


dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=15)
                for x in ['train', 'val']}

dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1)

# dset_classes = dsets['train'].classes

### SECTION 3 : Writing the functions that do training and validation phase. 

# These functions basically do forward propogation, back propogation, loss calculation, update weights of model, and save best model!


## The below function will train the model. Here's a short basic outline - 

# For the number of specified epoch's, the function goes through a train and a validation phase. Hence the nested for loop. 

# In both train and validation phase, the loaded data is forward propogated through the model (architecture defined ahead). 
# In PyTorch, the data loader is basically an iterator. so basically there's a get_element function which gets called everytime 
# the program iterates over data loader. So, basically, get_item on dset_loader below gives data, which contains 2 tensors - input and target. 
# target is the class number. Class numbers are assigned by going through the train/val folder and reading folder names in alphabetical order.
# So in our case cats would be first, dogs second and humans third class.

# Forward prop is as simple as calling model() function and passing in the input. 

# Variables are basically wrappers on top of PyTorch tensors and all that they do is keep a track of every process that tensor goes through.
# The benefit of this is, that you don't need to write the equations for backpropogation, because the history of computations has been tracked
# and pytorch can automatically differentiate it! Thus, 2 things are SUPER important. ALWAYS check for these 2 things. 
# 1) NEVER overwrite a pytorch variable, as all previous history will be lost and autograd won't work.
# 2) Variables can only undergo operations that are differentiable.

def train_model(model, criterion, optimizer, lr_scheduler, checkpoint_file, num_epochs=100, justval=False):
    since = time.time()
    print("##"*10)
    best_model = model

    # Loss history is saved below. Saved every epoch. 
    loss_history = {'train':[0], 'val':[0]}
    start_epoch  = 0
    best_top1    = 0

    if checkpoint_file:
        print()
        if os.path.isfile(checkpoint_file):
            try:
                checkpoint = torch.load(checkpoint_file)
                start_epoch = checkpoint['epoch']
                best_top1 = checkpoint['best_top1']
                loss_history = checkpoint['loss_history']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file, checkpoint['epoch']))
            except:
                print("Found the file, but couldn't load it.")
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))

    # params for gradient noise 
    gamma = .55

    for epoch in range(start_epoch, num_epochs):
        t = epoch - 30 
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        phases = ['train', 'val']
        if justval: 
            phases = ['val']
        for phase in phases:
            if phase == 'train':
                mode='train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode='val'

            losses = RunningAvg()
            epoch_acc_1 = RunningAvg()
            epoch_acc_5 = RunningAvg()

            counter=0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.float().cuda())
                    labels = Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                losses.update(loss.data[0], inputs.size(0))
                acc_top_1 = accuracy(outputs.data, labels.data)[0]
                epoch_acc_1.update(acc_top_1[0], inputs.size(0))

                if counter % 100==0:
                    print("It: {}, Loss: {:.4f}, Top 1: {:.4f}".format(counter, losses.avg, epoch_acc_1.avg))

                counter+=1
            # At the end of every epoch, tally up losses and accuracies
            time_elapsed = time.time() - since

            print_stats(epoch_num=epoch, train=mode, batch_time=time_elapsed, loss=losses, top1=epoch_acc_1)  

            loss_history[mode].append(losses.avg)
            is_best = epoch_acc_1.avg > best_top1
            best_top1 = max(epoch_acc_1.avg, best_top1)
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_top1': epoch_acc_1.avg,
                'loss_history': loss_history,
                'optimizer': optimizer.state_dict(),
                }, is_best)
            print('checkpoint saved!')

            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss',losses.avg,step=epoch)
                    foo.add_scalar_value('epoch_acc_1',epoch_acc_1,step=epoch)
                if epoch_acc_1.avg > best_top1:
                    best_top1= epoch_acc_1.avg
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_top1)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_top1))
    print('returning and looping back')
    return best_model

# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


## Helper Functions

def save_checkpoint(state, is_best, filename='../../checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../../model_best.pth.tar')
    

def print_stats(epoch_num=None, it_num=None, train=True, batch_time=None, loss=None, top1=None, top5=None): 
    progress_string = "Epoch %d" % epoch_num if epoch_num else ''
    if it_num is not None: 
        progress_string += ", Iteration %d" % it_num
    else: 
        progress_string += " finished"
    progress_string += ", Training set = %s\n" % (train)
    print(progress_string + 
          "\tLoss: {loss.avg:.4f}\n Accuracies: \n"
          "\tTop 1: {top1.avg:.3f}%\n".format(batch_time=batch_time, loss=loss, top1=top1))

def save(filename='trained_alexnet'):
    """Saves model using file numbers to make sure previous models are not overwritten"""
    filenum = 0
    while (os.path.exists(os.path.abspath('{}_v{}.pt'.format(filename, filenum)))):
        filenum += 1
    torch.save(model.state_dict(), '{}_v{}.pt'.format(filename, filenum))

### SECTION 4 : DEFINING MODEL ARCHITECTURE.

# We use Resnet18 here. If you have more computational power, feel free to swap it with Resnet50, Resnet100 or Resnet152.
# Since we are doing fine-tuning, or transfer learning we will use the pretrained net weights. In the last line, the number of classes has been specified.
# Set the number of classes in the config file by setting the right value for NUM_CLASSES.



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=BASE_LR)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Type 'tr' to train, 'test' to test, and 'metrics' to extract the error metrics'.  For 'test' and 'metrics' make sure to add another argument specifying the path of the model.")

    if sys.argv[1] == 'tr': 
        checkpoint_file = '../../checkpoint.pth.tar'
        # Run the functions and save the best model in the function model_ft.
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, checkpoint_file, num_epochs=100)

    elif sys.argv[1] == 'val': 
        checkpoint_file = '../../checkpoint.pth.tar'
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, checkpoint_file, num_epochs=100, justval=True)

    elif sys.argv[1] == 'test': 
        model_path = sys.argv[2] if len(sys.argv) > 2 else '../../checkpoint.pth.tar'
        checkpoint = torch.load(model_path)
        model_ft.load_state_dict(checkpoint['state_dict'])
        model_ft.eval()
        epoch_acc_1 = RunningAvg()
        counter=0

        values = []
        for i, data in enumerate(dset_loaders['test']):
            path, inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            model_label = np.argmax(outputs.data)
            actual_label = labels.data[0]
            values.append((path[0], actual_label, model_label))
            acc_top_1 = accuracy(outputs.data, labels.data)[0]
            if i % 100 == 0: 
                print(i)
            epoch_acc_1.update(acc_top_1[0], inputs.size(0))
        print("Test accuracy: {top1.avg:.3f}".format(top1=epoch_acc_1))

        with open('outputs.txt', 'w') as outfile: 
            for val in values:
                outfile.write("%s %s %s\n" % (str(val[0]), str(val[1]), str(val[2])))
