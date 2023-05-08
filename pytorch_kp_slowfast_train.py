import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
import os
import pickle
import sys


num_samples = 32
slow = num_samples // 4
batch_size = 4
num_label = 40

class Feeder(Dataset):
    # def __init__(self, data_path, label_path,
    #              random_choose=False, random_shift=False, random_move=False,
    #              window_size=-1, normalization=False, debug=False, use_mmap=True, random_mirror=False, random_mirror_p=0.5, is_vector=False):
    def __init__(self, data_path, label_path, debug=False, normalization=False, random_mirror=False, random_mirror_p=0.5, is_vector=False):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        # self.random_choose = random_choose
        # self.random_shift = random_shift
        # self.random_move = random_move
        # self.window_size = window_size
        self.normalization = normalization
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
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

        subset_indices_1 = np.random.choice(self.data.shape[2], size=num_samples, replace=False)
        # print(type(subset_indices), subset_indices.shape)
        data_fast = self.data[:, :, subset_indices_1, :, :]
        subset_indices_2 = np.random.choice(data_fast.shape[2], size=slow, replace=False)
        data_slow = data_fast[:, :, subset_indices_2, :, :]
        # print(data_fast.shape, data_slow.shape)
        self.data = [data_slow, data_fast]

        if self.debug:
            print(np.asarray(self.label).shape)
            print(np.asarray(self.data[0]).shape)
            print(np.asarray(self.data[1]).shape)
            print(np.asarray(self.sample_name).shape)
            print(self.sample_name[0])

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index]

        data_numpy_slow = self.data[0][index]
        data_numpy_slow = np.array(data_numpy_slow)
        data_numpy_fast = self.data[1][index]
        data_numpy_fast = np.array(data_numpy_fast)

        if self.normalization:
            assert data_numpy_slow.shape[0] == 3
            assert data_numpy_fast.shape[0] == 3
            if self.is_vector:
                data_numpy_slow[0,:,0,:] = data_numpy_slow[0,:,0,:] - data_numpy_slow[0,:,0,0].mean(axis=0)
                data_numpy_slow[1,:,0,:] = data_numpy_slow[1,:,0,:] - data_numpy_slow[1,:,0,0].mean(axis=0)
                data_numpy_fast[0,:,0,:] = data_numpy_fast[0,:,0,:] - data_numpy_fast[0,:,0,0].mean(axis=0)
                data_numpy_fast[1,:,0,:] = data_numpy_fast[1,:,0,:] - data_numpy_fast[1,:,0,0].mean(axis=0)
            else:
                data_numpy_slow[0,:,:,:] = data_numpy_slow[0,:,:,:] - data_numpy_slow[0,:,0,0].mean(axis=0)
                data_numpy_slow[1,:,:,:] = data_numpy_slow[1,:,:,:] - data_numpy_slow[1,:,0,0].mean(axis=0)
                data_numpy_fast[0,:,0,:] = data_numpy_fast[0,:,0,:] - data_numpy_fast[0,:,0,0].mean(axis=0)
                data_numpy_fast[1,:,0,:] = data_numpy_fast[1,:,0,:] - data_numpy_fast[1,:,0,0].mean(axis=0)

        return [data_numpy_slow, data_numpy_fast], label, index

# is_vector: false for joint
path = '/n/fs/scratch/yz7976/WLASL/start_kit/key_points_feature'
kp_type = 'joint'
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


# load slowfast model
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
model.blocks[6].proj = torch.nn.Linear(model.blocks[6].proj.in_features, num_label, bias=True) # modify the last layer and train it again
model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))


# Set to GPU or CPU
device = "cuda"
model = model.eval()
model = model.to(device)


criterion = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 20
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Define the learning rate schedule
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


best_val_acc = 0
loss_history = []
for epoch in range(num_epochs):
    print('current epoch: %d' % (epoch + 1))
    running_loss = 0.0
    for i, (data, label, index) in enumerate(data_loader_train, 0):
        data = [lbl.to(device) for lbl in data]
        label = label.to(device)
        # print(type(inputs), type(labels))
        # print(type(inputs[0]), type(labels[0]))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            loss_history.append(running_loss/10)
            running_loss = 0.0

    # update learning rate
    lr_scheduler.step()
    
    # validate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label, index in (data_loader_val):
            data = [lbl.to(device) for lbl in data]
            label = label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    val_acc = 100 * correct / total
    print('Validation accuracy: %d %%' % (val_acc))

    # save the best model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_kp_joint_slowfast.pth")

# Plot the loss history and save it to a file
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss_kp_joint_slowfast.png')

# Load the saved model
print('Loading the best model')
model.load_state_dict(torch.load("best_kp_joint_slowfast.pth"))

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data, label, index in (data_loader_test):
        data = [lbl.to(device) for lbl in data]
        label = label.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))