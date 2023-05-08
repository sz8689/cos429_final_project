import argparse
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torchvision import *
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
# from pytorch_kp_train_gpu_zsy import Feeder
from torch.utils.data import DataLoader
from pytorch_kp_slowfast_train_zsy import Feeder
from pytorch_classification_train_gpu import VideoDataset
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

path = '/n/fs/scratch/yz7976/WLASL/start_kit/key_points_feature'
with open(path + '/top_40_test_label.pkl', 'rb') as f:
    sample_test_name, test_label = pickle.load(f)

num_label = 40
device = "cuda"
num_samples = 32
slow = num_samples // 4
batch_size = 8 # change to 8 because rgb use less
num_label = 40

rgb_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
rgb_model.blocks[6].proj = torch.nn.Linear(2304, num_label, bias=True) # modify the last layer and train it again
rgb_model.to(device)
rgb_model.load_state_dict(torch.load("best_model_rgb_batch8_ep10.pth"))

# kp joint model
joint_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
joint_model.blocks[6].proj = torch.nn.Linear(joint_model.blocks[6].proj.in_features, num_label, bias=True) # modify the last layer and train it again
joint_model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
joint_model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# bone_model.fc = torch.nn.Linear(bone_model.fc.in_features, out_features=num_label, bias=True)
joint_model.to(device)
joint_model.load_state_dict(torch.load("kp_models/best_joint_ep15_rs.pth"))

# kp bone model
bone_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
bone_model.blocks[6].proj = torch.nn.Linear(bone_model.blocks[6].proj.in_features, num_label, bias=True) # modify the last layer and train it again
bone_model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
bone_model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# bone_model.fc = torch.nn.Linear(bone_model.fc.in_features, out_features=num_label, bias=True)
bone_model.to(device)
bone_model.load_state_dict(torch.load("kp_models/best_bone_ep15_rs.pth"))

# kp joint motion model
# joint_motion_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# joint_motion_model.blocks[6].proj = torch.nn.Linear(joint_motion_model.blocks[6].proj.in_features, num_label, bias=True) # modify the last layer and train it again
# joint_motion_model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# joint_motion_model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# # joint_motion_model.fc = torch.nn.Linear(joint_motion_model.fc.in_features, out_features=num_label, bias=True)
# joint_motion_model.to(device)
# joint_motion_model.load_state_dict(torch.load("kp_models/best_joint_motion_ep15_rs.pth"))

# # kp bone motion model
# bone_motion_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# bone_motion_model.blocks[6].proj = torch.nn.Linear(bone_motion_model.blocks[6].proj.in_features, num_label, bias=True) # modify the last layer and train it again
# bone_motion_model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# bone_motion_model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
# # joint_motion_model.fc = torch.nn.Linear(joint_motion_model.fc.in_features, out_features=num_label, bias=True)
# bone_motion_model.to(device)
# bone_motion_model.load_state_dict(torch.load("kp_models/best_bone_motion_ep15_rs.pth"))


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
# batch_size = 8
# num_videos = len(val_label) # 53
# num_channels = 3
# num_frames = 150
# num_batches = (num_videos + batch_size - 1) // batch_size
predictions = []

# # reshape and normalize
# joint_val_data = np.load(path + '/top_40_val_joint.npy')
# joint_val_data = joint_val_data.reshape(num_videos, num_channels, num_frames, -1)
# bone_val_data = np.load(path + '/top_40_val_bone.npy')
# bone_val_data = joint_val_data.reshape(num_videos, num_channels, num_frames, -1)
# # joint_motion_val_data = np.load(path + '/top_40_val_joint_motion.npy')
# # joint_motion_val_data = joint_motion_val_data.reshape(num_videos, num_channels, num_frames, -1)

# joint_val_data = do_normalization(joint_val_data)
# # joint_motion_val_data = do_normalization(joint_motion_val_data)
# bone_val_data = do_normalization(bone_val_data)
joint_model.eval()
bone_model.eval()
# joint_motion_model.eval()
# bone_motion_model.eval()
rgb_model.eval()

joint_val_feeder = Feeder(path+'/top_40_val_'+'joint'+'.npy', 
                        path+'/top_40_val_label.pkl',
                        debug=False,
                        normalization=True,
                        random_shift = True,
                        is_vector=False)

joint_data_loader_val= torch.utils.data.DataLoader(
                dataset=joint_val_feeder,
                batch_size=batch_size,
                shuffle=False)

bone_val_feeder = Feeder(path+'/top_40_val_'+'bone'+'.npy', 
                        path+'/top_40_val_label.pkl',
                        debug=False,
                        normalization=True,
                        random_shift = True,
                        is_vector=True)

bone_data_loader_val= torch.utils.data.DataLoader(
                dataset=bone_val_feeder,
                batch_size=batch_size,
                shuffle=False)

rgb_val_data = VideoDataset('../WLASL/start_kit/videos', split="val")
rgb_val_loader = DataLoader(rgb_val_data, batch_size=batch_size, shuffle=False)

# joint_motion_val_feeder = Feeder(path+'/top_40_val_'+'joint_motion'+'.npy', 
#                 path+'/top_40_val_label.pkl',
#                 debug=False,
#                 normalization=True,
#                 random_shift = True,
#                 is_vector=True)

# joint_motion_data_loader_val= torch.utils.data.DataLoader(
#                 dataset=joint_motion_val_feeder,
#                 batch_size=batch_size,
#                 shuffle=False)

# bone_motion_val_feeder = Feeder(path+'/top_40_val_'+'bone_motion'+'.npy', 
#                 path+'/top_40_val_label.pkl',
#                 debug=False,
#                 normalization=True,
#                 random_shift = True,
#                 is_vector=True)

# bone_motion_data_loader_val= torch.utils.data.DataLoader(
#                 dataset=bone_motion_val_feeder,
#                 batch_size=batch_size,
#                 shuffle=False)


# for i in range(num_batches):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, num_videos)
#     joint_batch = joint_val_data[start_idx:end_idx]
#     joint_batch = torch.from_numpy(joint_batch).float().to(device)
#     # joint_motion_batch = joint_motion_val_data[start_idx:end_idx]
#     # joint_motion_batch = torch.from_numpy(joint_motion_batch).float().to(device)
#     bone_batch = bone_val_data[start_idx:end_idx]
#     bone_batch = torch.from_numpy(bone_batch).float().to(device)
# for (joint_batch, _, _), (bone_batch, _, _), (joint_motion_batch, _, _), (bone_motion_batch, _, _) in zip(joint_data_loader_val, bone_data_loader_val, joint_motion_data_loader_val, bone_motion_data_loader_val):
for (joint_batch, _, _), (bone_batch, _, _), (rgb_batch, _) in zip(joint_data_loader_val, bone_data_loader_val, rgb_val_loader):
    # Move batches to device
    # joint_motion_batch = joint_motion_batch.to(device)
    joint_batch = [lbl.to(device) for lbl in joint_batch]
    bone_batch = [lbl.to(device) for lbl in bone_batch]
    # joint_motion_batch = [lbl.to(device) for lbl in joint_motion_batch]
    # bone_batch = bone_batch.to(device)
    # bone_motion_batch = [lbl.to(device) for lbl in bone_motion_batch]
    rgb_batch = [lbl.to(device) for lbl in rgb_batch]

    with torch.no_grad():
        outputs_joint = joint_model(joint_batch)
        outputs_bone = bone_model(bone_batch)
        outputs_rgb = rgb_model(rgb_batch)
        # outputs_joint_motion = joint_motion_model(joint_motion_batch)
        # outputs_bone_motion = bone_motion_model(bone_motion_batch)

    # batch_predictions = torch.stack([outputs_joint, outputs_bone, outputs_joint_motion, outputs_bone_motion], dim=1).cpu().numpy()
    batch_predictions = torch.stack([outputs_joint, outputs_bone, outputs_rgb], dim=1).cpu().numpy()

    predictions.append(batch_predictions)

predictions = np.concatenate(predictions, axis=0)

# # Define range for weights_joint
# weights_joint_range = np.linspace(0.1, 0.3, 11)

# # Define range for weights_bone
# weights_bone_range = np.linspace(0.7, 0.9, 11)

# Initialize variables to store best weights and best performance
best_weights = None
best_performance = -np.inf
num_iterations = 10000

# Loop over random combinations of weights
for i in range(num_iterations):
 # Generate random weights for each model
    weights_joint = random.uniform(0.4, 0.5)
    weights_bone = random.uniform(0.4, 0.5)
    weights_rgb = 1 - weights_joint - weights_bone

    # Compute weighted average of predictions
    ensemble_predictions = weights_joint * predictions[:, 0, :] + weights_bone * predictions[:, 1, :] + weights_rgb * predictions[:, 2, :]

    # Compute the predicted labels for the validation set
    val_pred_labels = np.argmax(ensemble_predictions, axis=1)
    # print(test_pred_labels)
    # print(test_label)
    val_accuracy = np.mean(val_pred_labels == val_label)

    # Update best weights and best performance
    if val_accuracy > best_performance:
        best_weights = (weights_joint, weights_bone, weights_rgb)
        best_performance = val_accuracy

# use the best weights for test set
joint_test_feeder = Feeder(path+'/top_40_test_'+'joint'+'.npy', 
                        path+'/top_40_test_label.pkl',
                        debug=False,
                        normalization=True,
                        random_shift = True,
                        is_vector=False)

joint_data_loader_test= torch.utils.data.DataLoader(
                dataset=joint_test_feeder,
                batch_size=batch_size,
                shuffle=False)

bone_test_feeder = Feeder(path+'/top_40_test_'+'bone'+'.npy', 
                        path+'/top_40_test_label.pkl',
                        debug=False,
                        normalization=True,
                        random_shift = True,
                        is_vector=True)

bone_data_loader_test= torch.utils.data.DataLoader(
                dataset=bone_test_feeder,
                batch_size=batch_size,
                shuffle=False)

# joint_motion_test_feeder = Feeder(path+'/top_40_test_'+'joint_motion'+'.npy', 
#                 path+'/top_40_test_label.pkl',
#                 debug=False,
#                 normalization=True,
#                 random_shift = True,
#                 is_vector=True)

# joint_motion_data_loader_test= torch.utils.data.DataLoader(
#                 dataset=joint_motion_test_feeder,
#                 batch_size=batch_size,
#                 shuffle=False)

# bone_motion_test_feeder = Feeder(path+'/top_40_test_'+'bone_motion'+'.npy', 
#                 path+'/top_40_test_label.pkl',
#                 debug=False,
#                 normalization=True,
#                 random_shift = True,
#                 is_vector=True)

# bone_motion_loader_test= torch.utils.data.DataLoader(
#                 dataset=bone_motion_test_feeder,
#                 batch_size=batch_size,
#                 shuffle=False)
rgb_test_data = VideoDataset('../WLASL/start_kit/videos', split="test")
rgb_test_loader = DataLoader(rgb_test_data, batch_size=batch_size, shuffle=False)

test_predictions = []
for (joint_batch, _, _), (bone_batch, _, _), (rgb_batch, _) in zip(joint_data_loader_test, bone_data_loader_test, rgb_test_loader):
    # Move batches to device
    # joint_motion_batch = joint_motion_batch.to(device)
    joint_batch = [lbl.to(device) for lbl in joint_batch]
    bone_batch = [lbl.to(device) for lbl in bone_batch]
    # joint_motion_batch = [lbl.to(device) for lbl in joint_motion_batch]
    # # bone_batch = bone_batch.to(device)
    # bone_motion_batch = [lbl.to(device) for lbl in bone_motion_batch]
    rgb_batch = [lbl.to(device) for lbl in rgb_batch]

    with torch.no_grad():
        outputs_joint = joint_model(joint_batch)
        outputs_bone = bone_model(bone_batch)
        outputs_rgb = rgb_model(rgb_batch)
        # outputs_joint_motion = joint_motion_model(joint_motion_batch)
        # outputs_bone_motion = bone_motion_model(bone_motion_batch)

    batch_predictions = torch.stack([outputs_joint, outputs_bone, outputs_rgb], dim=1).cpu().numpy()

    test_predictions.append(batch_predictions)

test_predictions = np.concatenate(test_predictions, axis=0)

ensemble_predictions = best_weights[0] * test_predictions[:, 0, :] + best_weights[1] * test_predictions[:, 1, :] + best_weights[2] * test_predictions[:, 2, :]
# # Compute the predicted labels for the test set
test_pred_labels = np.argmax(ensemble_predictions,axis=1)
# print(test_pred_labels)
# print(test_label)
acc = np.mean(test_pred_labels == test_label)

ensemble_probs = torch.nn.functional.softmax(torch.tensor(ensemble_predictions), dim=1)
ensemble_pred_classes = torch.argmax(ensemble_probs, dim=1)

p, r, f1, _ = precision_recall_fscore_support(test_label, ensemble_pred_classes, average=None)
cm = confusion_matrix(test_label, ensemble_pred_classes)
auc_roc = roc_auc_score(test_label, ensemble_probs, multi_class='ovr')
ap_scores = []

for i in range(40):
    target_i = (np.array(test_label) == i).astype(int)
    probs_i = ensemble_probs[:, i]
    ap_scores.append(average_precision_score(target_i, probs_i))

    # Compute the mean of the average precision scores over all classes to get the mAP score
map_score = np.mean(ap_scores)

# Open a text file for writing
with open('ensemble_keypoints_rgb_metrics.txt', 'w') as f:
    # Write each metric to the file
    f.write("Best weights:{}\n".format(best_weights))
    f.write("Accuracy: {}\n".format(acc))
    f.write("Precision: {}\n".format(p))
    f.write("Recall: {}\n".format(r))
    f.write("F1 score: {}\n".format(f1))
    f.write("Confusion matrix:\n")
    np.savetxt(f, cm, fmt='%d')
    f.write("\n")
    f.write("AUC-ROC: {}\n".format(auc_roc))
    f.write("MAP score: {}\n".format(map_score))