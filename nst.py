# Python's native libraries
import time
import os
import copy
from collections import defaultdict

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def train_photo(model, criterion, optimizer, photo, num_epochs=25):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = defaultdict(list)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            dataloaders = None
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = float(running_corrects) / dataset_sizes[phase]

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # metrics[phase + "_loss"].append(epoch_loss)
            # metrics[phase + "_acc"].append(epoch_acc)

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {best_acc:4f}')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model, metrics

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# img = mpimg.imread(os.getcwd() + "/data/content-images/tubingen.png")
# print(img)
# imgplot = plt.imshow(img)
# plt.show()
img = cv.imread(os.getcwd() + "/data/content-images/tubingen.png")
dim = (224, 224)
resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

model = torchvision.models.vgg19(pretrained=True)
# model.
model.eval()
model.features[3].register_forward_hook(get_activation('conv1_2'))

for param in model.parameters():
    param.requires_grad = False

output = model(torch.from_numpy(resized))
print(activation['conv1_2'])

#print(model)
# TODO Remove FC layers

model = model.to(device)

# TODO Implement needed loss
loss = None

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, finetuned_model.parameters()))