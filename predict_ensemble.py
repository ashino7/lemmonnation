import os
import pprint
import atexit
from argparse import ArgumentParser
import random
import timm
from tqdm import tqdm
import mlcrate as mlc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from collections import Counter, defaultdict
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from fastprogress import progress_bar, master_bar
from timm.models import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

from dataset import TestDataset
# from losses import lovasz_hinge
from augmentations.strong_aug import *
from augmentations.augmentation import *
from utils import find_exp_num, get_logger, remove_abnormal_exp, seed_everything, save_model

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-config", default='./config/predict.yaml')
    parser.add_argument("options", nargs="*")
    # args = parser.parse_args(args=['-config','config/base.yaml','options',''])
    args = parser.parse_args()
    return args


def load_state(model_name, model_path):
    model = eval(model_name)(False)
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_path)['weight'], strict=True)
        state_dict = torch.load(model_path)['weight']
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)['weight']
        state_dict = {k[7:] if k.startswith('weight.') else k: state_dict[k] for k in state_dict.keys()}
    return state_dict


def inference_ensemble(model, states, test_loader, device, transform):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        images = transform(images)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        add_preds = np.sum(avg_preds, axis=0)
        probs.append(add_preds)
    probs = np.concatenate(probs)
    return probs


def add_last_fc_layer(model, model_name, num_labels):
    if 'fc.weight' in model.state_dict().keys():
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    elif 'classifier.weight' in model.state_dict().keys():
        model.classifier = nn.Linear(model.classifier.in_features, num_labels)
    elif 'head.fc.weight' in model.state_dict().keys():
        model.head.fc = nn.Linear(model.head.fc.in_features, num_labels)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_labels)
    return model


def main():
    model_name1 = 'swsl_resnext101_32x4d'
    model_path1 = ['./colab_result/weight/exp_026/fold3_last_epoch.pth',
                   './colab_result/weight/exp_026/fold4_last_epoch.pth',
                   ]
    model_name2 = 'tf_efficientnet_b5_ns'
    model_path2 = ['./colab_result/weight/exp_019/fold0_last_epoch.pth',
                   './colab_result/weight/exp_019/fold1_last_epoch.pth',
                   './colab_result/weight/exp_019/fold2_last_epoch.pth',
                   ]

    test_path = './input/preprocess/v4/test_images'

    args = parse_args()
    config = OmegaConf.load(args.config)
    test_df = pd.read_csv(os.path.join(config.root, 'input/sample_submit.csv'), header=None, names=['id', 'class_num'])
    transform = eval(config.transform.name)(config.transform.size)

    X_test = test_df['id']
    test_data = TestDataset(X_test, transform['albu_val'], test_path)
    test_loader = DataLoader(test_data, **config.val_loader)

    # model1の準備
    states = []
    for i in range(len(model_path1)):
        states.append(load_state(model_name1, model_path1[i]))
    model = eval(model_name1)(True)
    model = add_last_fc_layer(model, model_name1, config.train.num_labels)
    predictions1 = inference_ensemble(model, states, test_loader, 'cuda', transform['torch_val'])

    # model1の準備
    states = []
    for i in range(len(model_path2)):
        states.append(load_state(model_name2, model_path2[i]))
    model = eval(model_name2)(True)
    model = add_last_fc_layer(model, model_name2, config.train.num_labels)
    predictions2 = inference_ensemble(model, states, test_loader, 'cuda', transform['torch_val'])

    predictions = predictions1 + predictions2
    predict_df = pd.DataFrame()
    predict_df['id'] = test_df['id']
    predict_df['label'] = predictions.argmax(1)
    predict_df.to_csv('predict.csv', index=False, header=False)


if __name__ == '__main__':
    main()
