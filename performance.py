import torch
import numpy as np
import pickle
import json
import os
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from pytorch_classification_train_gpu import VideoDataset
from pytorch_kp_slowfast_train import Feeder

rgb_flag = False

if rgb_flag:
    rgb_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    rgb_model.blocks[6].proj = torch.nn.Linear(model.blocks[6].proj.in_features, 40, bias=True) # modify the last layer and train it again
    rgb_model.load_state_dict(torch.load("best_model_rgb_batch8_ep10.pth"))
    rgb_model.to("cuda")
else:
    kp_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    kp_model.blocks[6].proj = torch.nn.Linear(kp_model.blocks[6].proj.in_features, 40, bias=True) # modify the last lay
    kp_model.blocks[5].pool[0] = torch.nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
    kp_model.blocks[5].pool[1] = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
    model_path = 'kp_models/best_joint_motion_ep15_rs.pth'
    kp_model.load_state_dict(torch.load(model_path))
    kp_model.to("cuda")


def calculate_metrics(model, device):
    model.eval()
    preds_list = []
    target_list = []
    probs_list = []
    if rgb_flag:
        test_data = VideoDataset('../WLASL/start_kit/videos', split="test")
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
        with torch.no_grad():
            for data in test_loader:
                videos, labels = data
                videos = [lbl.to(device) for lbl in videos]
                labels = labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)
                preds_list.append(predicted.cpu().numpy())
                target_list.append(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
            preds_all = np.concatenate(preds_list)
            target_all = np.concatenate(target_list)
            probs_all = np.concatenate(probs_list)
    else:
        path = '/n/fs/scratch/yz7976/WLASL/start_kit/key_points_feature'
        kp_type = 'joint_motion' # joint, joint_motion, bone, bone_motion
        
        # is_vector: false only for joint!!!!!!
        test_feeder = Feeder(path+'/top_40_test_'+kp_type+'.npy', 
                            path+'/top_40_test_label.pkl',
                            debug=False,
                            normalization=True,
                            random_shift = True,
                            is_vector=True)
        test_loader = DataLoader(
                dataset=test_feeder,
                batch_size=16,
                shuffle=True)
        with torch.no_grad():
            for data, labels, index in test_loader:
                data = [lbl.to(device) for lbl in data]
                labels = labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)
                preds_list.append(predicted.cpu().numpy())
                target_list.append(labels.cpu().numpy())
                probs_list.append(probs.cpu().numpy())
            preds_all = np.concatenate(preds_list)
            target_all = np.concatenate(target_list)
            probs_all = np.concatenate(probs_list)
    # print(preds_all)
    # print(target_all)
    # print(len(probs_all[0]))
    # print(probs_all[0].shape)
    # print(probs_all)

    acc = (preds_all == target_all).sum() / target_all.shape[0]
    p, r, f1, _ = precision_recall_fscore_support(target_all, preds_all, average=None)
    cm = confusion_matrix(target_all, preds_all)
    # # probs = torch.softmax(outputs, dim=1)[:, 1]
    # probs = torch.softmax(outputs, dim=1)
    # print(probs)
    # auc_roc = roc_auc_score(target_all, probs.cpu().numpy(), multi_class='ovr')
    auc_roc = roc_auc_score(target_all, probs_all, multi_class='ovr')
    # auc_pr = average_precision_score(target_all, probs.cpu().numpy())
    
    # Loop over each class and compute the average precision score for each class separately
    ap_scores = []
    for i in range(40):
        target_i = (target_all == i).astype(int)
        probs_i = probs_all[:, i]
        ap_scores.append(average_precision_score(target_i, probs_i))

    # Compute the mean of the average precision scores over all classes to get the mAP score
    map_score = np.mean(ap_scores)

    return acc, p, r, f1, cm, auc_roc, map_score

if rgb_flag:
    acc, p, r, f1, cm, auc_roc, map_score = calculate_metrics(rgb_model, "cuda")
else:
    acc, p, r, f1, cm, auc_roc, map_score = calculate_metrics(kp_model, "cuda")


# Open a text file for writing
with open('joint_motion_metrics.txt', 'w') as f:
    # Write each metric to the file
    f.write("Accuracy: {}\n".format(acc))
    f.write("Precision: {}\n".format(p))
    f.write("Recall: {}\n".format(r))
    f.write("F1 score: {}\n".format(f1))
    f.write("Confusion matrix:\n")
    np.savetxt(f, cm, fmt='%d')
    f.write("\n")
    f.write("AUC-ROC: {}\n".format(auc_roc))
    f.write("MAP score: {}\n".format(map_score))