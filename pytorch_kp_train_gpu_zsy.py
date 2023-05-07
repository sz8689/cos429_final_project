import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import time
import copy
import os
import pickle
import sys

class Feeder(Dataset):
    # def __init__(self, data_path, label_path,
    #              random_choose=False, random_shift=False, random_move=False,
    #              window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False):
    def __init__(self, data_path, label_path, debug=False, normalization=False, random_mirror=False, random_mirror_p=0.5, is_vector=False):
        """
        :param data_path: 
        :param label_path: 
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.normalization = normalization
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        if normalization:
            self.get_mean_map()
        # print(len(self.label))

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        self.data = np.load(self.data_path)
        # print(np.asarray(self.data).shape)
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2], -1)
        if self.debug:
            print(np.asarray(self.label).shape)
            print(np.asarray(self.data).shape)
            print(np.asarray(self.sample_name).shape)
            print(self.sample_name[0])

    def get_mean_map(self):
        data = self.data
        N, C, T, V = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 1, 3)).reshape((N * T, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[0] == 3
            data_numpy[0,:,:] = data_numpy[0,:,:] - data_numpy[0,:,0].mean(axis=0)
            data_numpy[1,:,:] = data_numpy[1,:,:] - data_numpy[1,:,0].mean(axis=0)

        return data_numpy, label, index


num_label = 40
batch_size = 8
n_epochs = 20
print_every = 10
learning_rate = 1e-3
weight_decay = 0.0001
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda"

path = '/n/fs/scratch/yz7976/WLASL/start_kit/key_points_feature'
kp_type = 'joint_motion'
train_feeder = Feeder(path+'/top_40_train_'+kp_type+'.npy', 
                      path+'/top_40_train_label.pkl',
                      debug=False,
                      normalization=True)
val_feeder = Feeder(path+'/top_40_val_'+kp_type+'.npy', 
                    path+'/top_40_val_label.pkl',
                      debug=False,
                      normalization=True)
test_feeder = Feeder(path+'/top_40_test_'+kp_type+'.npy', 
                     path+'/top_40_test_label.pkl',
                      debug=False,
                      normalization=True)

data_loader_train= torch.utils.data.DataLoader(
                dataset=train_feeder,
                batch_size=batch_size,
                shuffle=True)
data_loader_val= torch.utils.data.DataLoader(
                dataset=val_feeder,
                batch_size=batch_size,
                shuffle=True)
data_loader_test= torch.utils.data.DataLoader(
                dataset=test_feeder,
                batch_size=batch_size,
                shuffle=True)


net = models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(net.fc.in_features, out_features=num_label, bias=True)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay)


# valid_loss_min = np.Inf
best_val_acc = 0
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(data_loader_train)

for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    # for batch_idx, data_, target_, in enumerate(data_loader_train):
    for batch_idx, (data, label, index) in enumerate(data_loader_train):
    
        data, label = data.to(device), label.to(device)
        # optimizer.zero_grad()
        
        outputs = net(data)
        
        # Compute the loss with weight decay
        l2_regularization = torch.tensor(0.).to(device)
        for param in net.parameters():
            l2_regularization += torch.norm(param, p=2)**2
        loss = criterion(outputs, label) + weight_decay*l2_regularization
        # loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==label).item()
        total += label.size(0)
        # if (batch_idx) % 20 == 0:
        if (batch_idx) % print_every == 9:    # print every 10 mini-batches
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx + 1, total_step + 1, loss.item()))
        train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0


    with torch.no_grad():
        net.eval()
        for data, label, index in (data_loader_val):
            data, label = data.to(device), label.to(device)
            outputs_t = net(data)
            loss_t = criterion(outputs_t, label)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==label).item()
            total_t += label.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(data_loader_val))
        # network_learned = batch_loss < valid_loss_min
        network_learned = (100 * correct_t/total_t) > best_val_acc
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            # valid_loss_min = batch_loss
            best_val_acc = (100 * correct_t/total_t)
            torch.save(net.state_dict(), 'best_kp_joint_motion_model_batch8_ep20_wd.pt')
            print('Improvement-Detected, save-model')
    net.train()

# Plot the loss history and save it to a file
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss_gpu40_kp_joint_motion_batch8_ep20_wd.png')

# Load the saved model
print('Loading the best model')
net.load_state_dict(torch.load("best_kp_joint_motion_model_batch8_ep20_wd.pt"))

# Evaluate the model on the test set
with torch.no_grad():
        net.eval()
        for data, label, index in (data_loader_test):
            data, label = data.to(device), label.to(device)
            outputs_t = net(data)

            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==label).item()
            total_t += label.size(0)
        print(f'Test acc: {(100 * correct_t/total_t):.4f}\n')