import cv2
import torchvision.transforms as transforms
import torch
from torchvision.transforms import Compose, Lambda, RandomCrop
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import matplotlib.pyplot as plt
import pickle
import json

## CHANGE best_model name every time before running the script

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

# set up 
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 224
num_frames = 32
slowfast_alpha = 4
num_clips = 10
num_crops = 3

frame_transform = transforms.Compose([
    transforms.ToTensor()
])

video_transform = transforms.Compose([
    UniformTemporalSubsample(num_frames),
    ShortSideScale( 
        size=side_size
    ),
    # CenterCropVideo(crop_size),
    RandomCrop(crop_size),
    Lambda(lambda x: x/255.0), # normalization
    NormalizeVideo(mean, std),
    PackPathway()
]) 

class VideoDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        exclude_dir_name = ['train', 'test', 'validation'] # exclude these directories
        # self.labels = [label for label in os.listdir('../WLASL/start_kit/videos') if label not in exclude_dir_name]
        
        with open('../WLASL/start_kit/key_points_feature/top_40_{}_label.pkl'.format(split), 'rb') as f:
            sample_names, labels = pickle.load(f)

        with open('../WLASL/start_kit/key_points_feature/top_40_label_to_index.json') as f:
            self.label_map = json.load(f)
        label_map_r = {v: k for k, v in self.label_map.items()}
        
        self.data = []
        for i, label in enumerate(labels):
            label_dir = os.path.join(data_dir, label_map_r[label])
            video_path = os.path.join(label_dir, sample_names[i] + ".mp4")
            self.data.append((video_path, label_map_r[label]))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]

        # process video to each frame
        video = cv2.VideoCapture(video_path)
        video_data = []
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(frame_transform(frame))
        video.release()
        
        # transform the video
        video_tensor = torch.stack(video_data).float()
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        video_tensor = video_transform(video_tensor)
        
        # label_idx = self.label_to_idx[label]
        label_idx = self.label_map[label]
        return video_tensor, label_idx

def train_model():
    video_dir = '../WLASL/start_kit/videos'
    train_data = VideoDataset(video_dir, split="train")
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    val_data = VideoDataset(video_dir, split="val")
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True)

    test_data = VideoDataset(video_dir, split="test")
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model.blocks[6].proj = torch.nn.Linear(2304, 40, bias=True) # modify the last layer and train it again

    # Set to GPU or CPU
    device = "cuda"
    model = model.to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, len(train_data.labels))

    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    num_epochs = 10
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Define the learning rate schedule
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0
    train_acc_history = []
    val_acc_history = []
    loss_history = []
    for epoch in range(num_epochs):
        print('current epoch: %d' % (epoch + 1))
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs and labels from the data loader
            inputs, labels = data
            # print(type(inputs), type(labels))
            # print(type(inputs[0]), type(labels[0]))
            inputs = [lbl.to(device) for lbl in inputs]
            labels = labels.to(device)
            # print(type(inputs), type(labels))
            # print(type(inputs[0]), type(labels[0]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                loss_history.append(running_loss/10)
                running_loss = 0.0

        train_acc = 100 * correct / total
        print('Train accuracy: %d %%' % (train_acc))
        train_acc_history.append(train_acc)

        # update learning rate
        lr_scheduler.step()
        
        # validate the model
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                videos, labels = data
                videos = [lbl.to(device) for lbl in videos]
                labels = labels.to(device)

                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print('Validation accuracy: %d %%' % (val_acc))
        val_acc_history.append(val_acc)

        # save the best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_rgb_batch8_ep10.pth")
    # save data
    with open("training_loss_rgb_batch8_ep10.txt", "w") as output:
        output.write(str(loss_history))
    with open("train_acc_rgb_batch8_ep10.txt", "w") as output:
        output.write(str(train_acc_history))
    with open("val_acc_rgb_batch8_ep10.txt", "w") as output:
        output.write(str(val_acc_history))

    # Plot the loss history and save it to a file
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss_rgb_batch8_ep10.png')


    # Load the saved model
    print('Loading the best model')
    model.load_state_dict(torch.load("best_model_rgb_batch8_ep10.pth"))

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            videos, labels = data
            videos = [lbl.to(device) for lbl in videos]
            labels = labels.to(device)

            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
