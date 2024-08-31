import torch
from torch import nn, optim
from torchvision import models
import argparse
from PIL import Image
import numpy as np
import json
from collections import OrderedDict
from time import time

class Util:
    @staticmethod
    def load_data(data_dir):
        # Load data and transform
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)

        # Create dataloaders
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

        return trainloader, validloader, testloader

    @staticmethod
    def build_model(hidden_units, arch='densenet121', learning_rate=0.001):
        if arch == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif arch == 'vgg13':
            model = models.vgg13(pretrained=True)
        else:
            raise ValueError('Model architecture not supported')

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        return model, criterion, optimizer

    @staticmethod
    def test_accuracy(model, testloader):
        model.eval()
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                equality = (predicted == labels).float()
                accuracy += equality.mean()
        return accuracy

    @staticmethod
    def train_model(model, trainloader, validloader, criterion, optimizer, epochs):
        model.to(device)
        steps = 0
        print_every = 40
        running_loss = 0
        for epoch in range(epochs):
            model.train()
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    with torch.no_grad():
                        validation_accuracy = Util.test_accuracy(model, validloader)
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation accuracy: {validation_accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

    @staticmethod
    def save_model(model, save_path, train_data):
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {
            'model': model,
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load_model(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        return model

    @staticmethod
    def process_image(image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform(image)

    @staticmethod
    def predict(image_path, model, topk=5):
        model.to(device)
        model.eval()
        image = Util.process_image(image_path)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
        probabilities, indices = torch.topk(output, topk)
        probabilities = probabilities.exp().cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in indices]
        return probabilities, classes

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Data directory path')
parser.add_argument('--save_dir', type=str, default='./', help='Directory to save the checkpoint')
parser.add_argument('--arch', type=str, default='densenet121', help='Model architecture (densenet121 or vgg13)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=
