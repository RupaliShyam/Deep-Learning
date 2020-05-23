import torch
import torch.nn as nn
import os
import cv2
from torchvision import models
from skimage.transform import resize
import numpy as np



class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, tf):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        folder = 0
        data = []
        for dir in os.listdir(self.root_dir):
            category_folder = os.path.join(self.root_dir, dir)
            for file in os.listdir(category_folder):
                filepath = os.path.join(category_folder, file)
                # image = io.imread(filepath)
                img = cv2.imread(filepath)
                if(tf == True):
                    img = resize(img, (224, 224))
                    img = np.moveaxis(img, 2, 0)
                image = torch.from_numpy(img)
                sample = (image, folder)
                data.append(sample)
                # return sample
            folder += 1
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]
        return sample


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.fc1(x.reshape(-1, 32 * 32 * 3))
        out = self.tanh(out)
        out = self.fc2(out)
        return out

def training(model, train_loader):
    #writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            '''
            if (i + 1) % 20 == 0:
                #writer.add_scalar("Training Loss", loss.item(), i)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
            '''
        if(epoch + 1) % 4 == 0:
            print('Epoch [{}/{}] Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    return model


def evaluation(model, test_loader):
    #writer = SummaryWriter()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #writer.add_scalar("Accuracy", (100 * correct / total), i)

    return (100 * correct / total)

train_dataset = CifarDataset("cifar10_train", tf=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

test_dataset = CifarDataset("cifar10_test", tf=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False)
##################################################################

input_size = 3072
num_classes = 10
num_epochs = 20
#batch_size = 100  # Batch gradient descent
hidden_size = 2800
learning_rate = 0.00001
device = 'cuda:0'


model = MultilayerPerceptron(input_size, hidden_size, num_classes).to(device)
trained_model = training(model, train_dataloader)
train_score = evaluation(trained_model, train_dataloader)
print('Accuracy of the network on the 50000 train images: {} %'.format(train_score))
test_score = evaluation(trained_model, test_dataloader)
print('Accuracy of the network on the 10000 test images: {} %'.format(test_score))

#hidden_layer = [1000, 1540, 2048]
#learning_rate = [0.00001, 0.0001, 0.001]
#optimizer = ['Adam', 'SGD', 'RMSprop']
#Regularizer = [0.01, 0.1, 0.001]

#Transfer Learning

#Feature extraction --> param.requires_grad = False
def train_model(model, dataloaders, num_epochs, feature_extract):
    criterion = nn.CrossEntropyLoss()
    optimizer = optimze(model_ft, feature_extract)
    model.train()
    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(dataloaders):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(dataloaders), loss.item()))
    return model

def set_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract):

    model = models.mobilenet_v2(pretrained=True)
    set_requires_grad(model, feature_extract)
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(features, num_classes)


    return model

def optimze(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    return optimizer_ft

train_dataset = CifarDataset("cifar10_train", tf=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128,
                                               shuffle=True)

test_dataset = CifarDataset("cifar10_test", tf=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False)

#Feature Extraction
print("Transfer Learning using Feature Extraction")
model_ft = initialize_model(num_classes, feature_extract=True)
model_ft = model_ft.to(device)
trained_model_ft = train_model(model_ft, train_dataloader, num_epochs=num_epochs, feature_extract=True)
train_score = evaluation(trained_model_ft, train_dataloader)
print('Accuracy of the network on the 50000 train images: {} %'.format(train_score))
test_score = evaluation(trained_model_ft, test_dataloader)
print('Accuracy of the network on the 10000 test images: {} %'.format(test_score))

#Fine Tuning
print("Transfer Learning using Fine Tuning")
model_ft = initialize_model(num_classes, feature_extract=False)
model_ft = model_ft.to(device)
trained_model_ft = train_model(model_ft, train_dataloader, num_epochs=num_epochs, feature_extract=False)
train_score = evaluation(trained_model_ft, train_dataloader)
print('Accuracy of the network on the 50000 train images: {} %'.format(train_score))
test_score = evaluation(trained_model_ft, test_dataloader)
print('Accuracy of the network on the 10000 test images: {} %'.format(test_score))




