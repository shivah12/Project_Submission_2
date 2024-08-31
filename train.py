import numpy as np
import os
import torch
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from time import time

# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str,
                    help='Provide the data directory, mandatory')
parser.add_argument('--save_dir', type=str, default='./',
                    help='Provide the save directory')
parser.add_argument('--arch', type=str, default='densenet121',
                    help='densenet121 or vgg13')
# Hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate, default value 0.001')
parser.add_argument('--hidden_units', type=int, default=512,
                    help='Number of hidden units. Default value is 512')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs')
# GPU
parser.add_argument('--gpu', action='store_true',
                    help="Add to activate CUDA")

# Setting values for data loading
args_in = parser.parse_args()

if args_in.gpu:
    device = torch.device("cuda")
    print("****** CUDA activated ********************")
else:
    device = torch.device("cpu")

# Define data transforms for training, validation, and testing
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(args_in.data_dir, x), data_transforms[x])
                  for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'valid', 'test']}

# Define label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build the model
print("------ building the model ----------------")

layers = args_in.hidden_units
learning_rate = args_in.learning_rate

if args_in.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features
elif args_in.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    in_features = model.classifier[0].in_features
else:
    raise ValueError('Model arch error')

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(in_features, layers),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(layers, 102),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)

print("****** model arch:", args_in.arch)
print("------ model building finished -----------")

# Training the model
print("------ training the model ----------------")

epochs = args_in.epochs
steps = 0
running_loss = 0
print_every = 10
for epoch in range(epochs):
    t1 = time()
    for inputs, labels in dataloaders['train']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {test_loss / len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()
    t2 = time()
    print("Elapsed Runtime for epoch {}: {}s.".format(epoch + 1, t2 - t1))

print("------ model training finished -----------")

# Testing the model
print("------ test the model --------------------")
model.to(device)
model.eval()
accuracy = 0

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")
model.train()
print("------ model testing finished ------------")

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'class_to_idx': model.class_to_idx,
    'model_state_dict': model.state_dict(),
    'classifier': model.classifier,
    'arch': args_in.arch
}

save_path = os.path.join(args_in.save_dir, 'checkpoint.pth')
torch.save(checkpoint, save_path)
print("------ model saved -----------------------")
