import argparse
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torchvision import *
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
# from pytorch_kp_train_gpu_zsy import Feeder
# from torch.utils.data import Dataset, DataLoader

path = '/n/fs/scratch/yz7976/WLASL/start_kit/key_points_feature'
with open(path + '/top_40_test_label.pkl', 'rb') as f:
    sample_test_name, test_label = pickle.load(f)

def do_normalization(data_numpy):
    for i in range(data_numpy.shape[0]):
        data_numpy[i, 0, :, :] = data_numpy[i, 0, :, :] - data_numpy[i, 0, :, 0].mean(axis=0)
        data_numpy[i, 1, :, :] = data_numpy[i, 1, :, :] - data_numpy[i, 1, :, 0].mean(axis=0)
    return data_numpy

from sklearn.linear_model import LogisticRegression

def compute_blending_weights(val_labels, val_predictions):
    # Flatten the validation set predictions into a 2D array
    num_val_videos, num_models, num_classes = val_predictions.shape
    X = val_predictions.reshape(num_val_videos, num_models*num_classes)
    # print(val_label.shape)
    print(X.shape)
    print(val_labels.shape)

    # Flatten the validation set labels into a 1D array
    # y = np.argmax(val_labels, axis=1)

    # # Fit a logistic regression model to the validation set predictions
    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    # clf.fit(X, y)

    # lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, val_labels)
    # lr.fit(X, val_labels)
    blending_weights = ridge.coef_
    # blending_weights = clf.coef_
    # blending_weights = lr.coef_
    print(blending_weights.shape)

    # # Compute the blending weights by averaging the coefficients across models
    # blending_weights = clf.coef_.reshape(num_models, -1)
    # blending_weights = blending_weights.reshape((num_models, num_classes, num_classes))
    # print(blending_weights.shape)
    # blending_weights = np.mean(blending_weights, axis=2)
    # # blending_weights = np.mean(blending_weights, axis=2)
    # print(blending_weights.shape)

    # Return the learned logistic regression coefficients as blending weights
    return blending_weights

# rgb model
# r1 = open('./joint_epoch_222_0.4561.pkl', 'rb')
# r1 = list(pickle.load(r1).items())
num_label = 40
device = "cuda"

# rgb_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# rgb_model.blocks[6].proj = torch.nn.Linear(2304, num_label, bias=True) # modify the last layer and train it again
# rgb_model.load_state_dict(torch.load("best_model_gpu40_batch16_Adam_ep60_split.pth"))

# kp joint model
joint_model = models.resnet18(pretrained=True)
joint_model.fc = torch.nn.Linear(joint_model.fc.in_features, out_features=num_label, bias=True)
joint_model.to(device)
joint_model.load_state_dict(torch.load("best_kp_joint_model_batch8_ep20_wd.pt"))

# kp bone model
bone_model = models.resnet18(pretrained=True)
bone_model.fc = torch.nn.Linear(bone_model.fc.in_features, out_features=num_label, bias=True)
bone_model.to(device)
bone_model.load_state_dict(torch.load("best_kp_bone_model_batch8_ep20_wd.pt"))

# kp joint motion model
joint_motion_model = models.resnet18(pretrained=True)
joint_motion_model.fc = torch.nn.Linear(joint_motion_model.fc.in_features, out_features=num_label, bias=True)
joint_motion_model.to(device)
joint_motion_model.load_state_dict(torch.load("best_kp_joint_motion_model_batch8_ep20_wd.pt"))

# # kp bone motion model
# bone_motion_model = models.resnet18(pretrained=True)
# bone_motion_model.load_state_dict(torch.load("best_kp_bone_motion_model_batch8_ep20_wd.pt"))


# kp model
# r2 = open('./bone_epoch_237_0.4327.pkl', 'rb')
# r2 = list(pickle.load(r2).items())

# r3 = open('./joint_motion_epoch_245_0.2723.pkl', 'rb')
# r3 = list(pickle.load(r3).items())
# r4 = open('./bone_motion_epoch_241_0.3126.pkl', 'rb')
# r4 = list(pickle.load(r4).items())

# get the validation result for each model
with open(path + '/top_40_val_label.pkl', 'rb') as f:
    sample_val_name, val_label = pickle.load(f)

# alpha = [0.7, 0.3] # for joint and joint motion first
# alpha = [1.0,0.9,0.5,0.5] # 51.50
batch_size = 8
num_videos = len(val_label) # 53
num_channels = 3
num_frames = 150
num_batches = (num_videos + batch_size - 1) // batch_size
predictions = []

# reshape and normalize
joint_val_data = np.load(path + '/top_40_val_joint.npy')
joint_val_data = joint_val_data.reshape(num_videos, num_channels, num_frames, -1)
bone_val_data = np.load(path + '/top_40_val_bone.npy')
bone_val_data = joint_val_data.reshape(num_videos, num_channels, num_frames, -1)
# joint_motion_val_data = np.load(path + '/top_40_val_joint_motion.npy')
# joint_motion_val_data = joint_motion_val_data.reshape(num_videos, num_channels, num_frames, -1)

joint_val_data = do_normalization(joint_val_data)
# joint_motion_val_data = do_normalization(joint_motion_val_data)
bone_val_data = do_normalization(bone_val_data)
joint_model.eval()
bone_model.eval()

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_videos)
    joint_batch = joint_val_data[start_idx:end_idx]
    joint_batch = torch.from_numpy(joint_batch).float().to(device)
    # joint_motion_batch = joint_motion_val_data[start_idx:end_idx]
    # joint_motion_batch = torch.from_numpy(joint_motion_batch).float().to(device)
    bone_batch = bone_val_data[start_idx:end_idx]
    bone_batch = torch.from_numpy(bone_batch).float().to(device)

    with torch.no_grad():
        outputs_joint = joint_model(joint_batch)
        outputs_bone = bone_model(bone_batch)

        # pred_prob_joint, pred_joint = torch.max(outputs_joint, dim=1)
        # pred_prob_joint_motion, pred_joint_motion = torch.max(outputs_joint_motion, dim=1)
        # pred_1 = model_1(batch)
        # pred_2 = model_2(batch)
        # pred_3 = model_3(batch)
        # pred_4 = model_4(batch)
        # pred_5 = model_5(batch)
    batch_predictions = torch.stack([outputs_joint, outputs_bone], dim=1).cpu().numpy()

    predictions.append(batch_predictions)

predictions = np.concatenate(predictions, axis=0)

# Compute the blending weights based on the validation set
val_labels = np.zeros((len(val_label), num_label))
for i, l in enumerate(val_label):
    val_labels[i, l] = 1

blending_weights = compute_blending_weights(val_labels, predictions)
print(blending_weights)

# # Compute the final predictions for the test set

# final_predictions = np.average(test_label, axis=1, weights=blending_weights) # [num_test_videos, num_classes]

# reshape and normalize
test_num_videos = len(test_label)
joint_test_data = np.load(path + '/top_40_test_joint.npy')
joint_test_data = joint_test_data.reshape(test_num_videos, num_channels, num_frames, -1)
bone_test_data = np.load(path + '/top_40_test_bone.npy')
bone_test_data = bone_test_data.reshape(test_num_videos, num_channels, num_frames, -1)
joint_motion_test_data = np.load(path + '/top_40_test_joint_motion.npy')
joint_motion_test_data = joint_motion_test_data.reshape(test_num_videos, num_channels, num_frames, -1)

joint_test_data = do_normalization(joint_test_data)
bone_test_data = do_normalization(bone_test_data)
joint_motion_test_data = do_normalization(joint_motion_test_data)
test_predictions = []
test_num_batches = (test_num_videos + batch_size - 1) // batch_size
for i in range(test_num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, test_num_videos)
    joint_batch = joint_test_data[start_idx:end_idx]
    joint_batch = torch.from_numpy(joint_batch).float().to(device)
    joint_motion_batch = joint_motion_test_data[start_idx:end_idx]
    joint_motion_batch = torch.from_numpy(joint_motion_batch).float().to(device)
    bone_batch = bone_test_data[start_idx:end_idx]
    bone_batch = torch.from_numpy(bone_batch).float().to(device)

    with torch.no_grad():
        outputs_joint = joint_model(joint_batch)
        outputs_joint_motion = joint_motion_model(joint_motion_batch)
        outputs_bone = bone_model(bone_batch)

        # pred_prob_joint, pred_joint = torch.max(outputs_joint, dim=1)
        # pred_prob_joint_motion, pred_joint_motion = torch.max(outputs_joint_motion, dim=1)
        # pred_1 = model_1(batch)
        # pred_2 = model_2(batch)
        # pred_3 = model_3(batch)
        # pred_4 = model_4(batch)
        # pred_5 = model_5(batch)
    batch_predictions = torch.stack([outputs_joint, outputs_joint_motion, outputs_bone], dim=1).cpu().numpy()

    test_predictions.append(batch_predictions)

test_predictions = np.concatenate(test_predictions, axis=0)

ensemble_predictions = 0.5 * test_predictions[:, 0, :] + 0.3 * test_predictions[:, 1, :] + 0.2 * test_predictions[:, 2, :]

# num_test_videos, num_models, num_classes = test_predictions.shape
# test_predictions_flat = test_predictions.reshape(num_test_videos, num_models*num_classes)
# ensemble_predictions = np.dot(test_predictions_flat, blending_weights.T)

# # Compute the predicted labels for the test set
test_pred_labels = np.argmax(ensemble_predictions,axis=1)
# print(test_pred_labels)
# print(test_label)
test_accuracy = np.mean(test_pred_labels == test_label)
# Compute the accuracy of the predicted labels
print(f'Test acc: {test_accuracy}\n')