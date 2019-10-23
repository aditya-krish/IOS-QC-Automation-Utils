# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:13:47 2019

@author: Aditya
Train NN on the scan images
"""

import torch
import torchvision
from torch import nn, optim
from torchvision.models import densenet121 as my_model
from torchvision import transforms
from torch.utils import data
import time
import copy
import matplotlib.pyplot as plt
import math

torch.manual_seed(10)
torch.cuda.empty_cache()

train_root = r"E:\bite_train_root"
test_root = r"E:\bite_issue\i am unsure"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([transforms.Resize([224,224]),
                                     transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor()                              
#                                ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

model = my_model(pretrained=True)
NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS = 2, 5, 1
classifier_input_size = model.classifier.in_features
model.classifier = nn.Linear(in_features = classifier_input_size
                             ,out_features = NUM_CLASSES
                             ,bias=True)

dataset = torchvision.datasets.ImageFolder(root = train_root,
                                                 transform = data_transform
                                                 )

train_dataset, test_dataset = torch.utils.data.random_split(
                                dataset,
                                [math.floor(0.8*(len(dataset))), 
                                 math.ceil(0.2*len(dataset))])
# alternate implementation to test on pre-designed dataset
'''
train_dataset = dataset

test_dataset = torchvision.datasets.ImageFolder(root = train_root,
                                                 transform = data_transform
                                                 )
'''

train_data_loader = data.DataLoader(train_dataset,
                                    batch_size = BATCH_SIZE,
                                    num_workers=0,
                                    shuffle=True
                                    )

test_data_loader = data.DataLoader(test_dataset,
                                   batch_size = 6,
                                   num_workers=0,
                                   shuffle=False
                                   )

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
model.to(device)
store_batch_loss = []
store_epoch_loss = []
best_acc = 0.0
since_training_start = time.time()

for _ in range(NUM_EPOCHS):
    
    running_loss = 0.0
    running_corrects = 0
    since_epoch_start = time.time()
    for inputs, targets in train_data_loader:
        
        torch.cuda.empty_cache()
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        prediction = torch.argmax(outputs,dim=1)
        print('predictions:',prediction.to('cpu'))
        print('targets    :',targets.to('cpu'))
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(prediction==targets.data)
        store_batch_loss.append(loss.item())
        
    store_epoch_loss.append(running_loss)
    time_for_epoch = time.time() - since_epoch_start
    epoch_accuracy = running_corrects.double()/len(train_data_loader.dataset)
    if epoch_accuracy>best_acc:
        best_model = copy.deepcopy(model)
        best_model.to('cpu')
        print('Model has improved after this epoch')
        best_acc = epoch_accuracy
    else:
        print('Model has not improved after this epoch')
    print(f'time taken to train this epoch is: {time_for_epoch:5.1f} secs to achieve a best accuracy of {100*best_acc: 2.1f}')

time_for_training = time.time() - since_training_start
print(f'Training completed in {time_for_training:.1f} secs')        

plt.figure(figsize=(9,9))
plt.title('Evolution of batch loss during training') 
plt.plot(store_batch_loss)
plt.xlabel('Number of batches')
plt.ylabel('Batch loss')
plt.show()   
    
plt.figure(figsize=(9,9))
plt.title('Evolution of total loss during training') 
plt.plot(store_epoch_loss)
plt.xlabel('Number of epochs')
plt.ylabel('Epoch loss')
plt.show()  
testing_corrects = 0 
    
for inputs, targets in test_data_loader:
    outputs = best_model(inputs)
    preds = torch.argmax(outputs,dim=1)
    testing_corrects += torch.sum(preds==targets.data)
    # plotting in a grid
    fig = plt.figure(figsize=(9,9))
    for x in range(len(inputs[:,1,1,1])):
        ax = fig.add_subplot(3,2,x+1)
        ax.imshow(inputs[x,:,:,:].permute(1,2,0))
        ax.axis('off')
        ax.set_title(f'correct label: {targets[x]}, prediction: {preds[x]}',fontsize=10)
    plt.show()
    
testing_accuracy = testing_corrects.double()/len(test_data_loader.dataset)
print(f'The accuracy on the dev-test set is {100*testing_accuracy:2.1f}')