import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

class VideoDataset(Dataset):
    def __init__(self, data_dir, split="train", test_size=0.1, val_size=0.1, random_state=429):
        self.labels = os.listdir(data_dir)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.data = []
        for label in self.labels:
            label_dir = os.path.join(data_dir, label)
            video_files = os.listdir(label_dir)
            video_paths = [os.path.join(label_dir, video_file) for video_file in video_files]

            train_val_paths, test_paths = train_test_split(video_paths, test_size=test_size, random_state=random_state)
            train_paths, val_paths = train_test_split(train_val_paths, test_size=val_size/(1-test_size), random_state=random_state)

            # each label will have 80% go to training, 10% go to validation, and 10% go to test
            if split == "train":
                self.data += [(video_path, label) for video_path in train_paths]
            elif split == "val":
                self.data += [(video_path, label) for video_path in val_paths]
            elif split == "test":
                self.data += [(video_path, label) for video_path in test_paths]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        # process video to each frame
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize the frame
            frame = cv2.resize(frame, (224, 224))
            frame = transforms.ToTensor()(frame)
            frames.append(frame)
        video.release()
        
        print(frames)
#         # Pad or crop the frames to a fixed size
#         num_frames = len(frames)
#         if num_frames < 16:
#             # Pad the frames with zeros
#             pad_frames = [torch.zeros((3, size, size)) for i in range(16 - num_frames)]
#             frames = frames + pad_frames
#         elif num_frames > 16:
#             # Crop the frames to the first 16 frames
#             frames = frames[:16]

        # Stack the frames into a single tensor
        # frames = torch.stack(frames)
        # video_tensor = torch.tensor(frames, dtype=torch.float32)
        # video_tensor = video_tensor.permute(3, 0, 1, 2)
        video_tensor = torch.stack(frames)
        # video_tensor size: [num_frames, num channel, 224, 244]
        # video_tensor = video_tensor.view(-1, 3, 224, 224)
        # print(f"getitem: {video_tensor.shape}")
        label_idx = self.label_to_idx[label]
        return video_tensor, label_idx

video_dir = '../WLASL/start_kit/videos'
train_data = VideoDataset(video_dir, split="train")
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

val_data = VideoDataset(video_dir, split="val")
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

test_data = VideoDataset(video_dir, split="test")
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_data.labels))

criterion = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_val_acc = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs and labels from the data loader
        inputs, labels = data
        # problem is in here, the expected dimension is [batch_size, num_channel, height, width]
        # now input is [4, 16, 3, 224, 224] -> how to conver it to [4, 3, 224, 224]?
        inputs = inputs.view(-1, 3, 224, 224)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # print(f"epoch: {inputs.shape}")
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # validate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print('Validation accuracy: %d %%' % (val_acc))

    # save the best model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
            
# Load the saved model
print('Loading the best model')
model.load_state_dict(torch.load("best_model.pth"))

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))