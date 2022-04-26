import os
import pprint
import atexit
from argparse import ArgumentParser
import random
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
from augmentations.augmentation import met_transform1
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


def inference(model, states, test_loader, device, transform):
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
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def main():
    model_name = ['tf_efficientnet_b5_ns',
                  ]
    model_path = ['./colab_result/weight/exp_019/fold0_last_epoch.pth',
                  './colab_result/weight/exp_019/fold1_last_epoch.pth',
                  './colab_result/weight/exp_019/fold2_last_epoch.pth',
                  ]
    # model_path = ['./weight/exp_004/fold0_best_loss.pth',
    #               './weight/exp_004/fold1_best_loss.pth',
    #               './weight/exp_004/fold2_best_loss.pth',
    #               ]
    test_path = './input/preprocess/test_images_2'

    args = parse_args()
    config = OmegaConf.load(args.config)
    test_df = pd.read_csv(os.path.join(config.root, 'input/sample_submit_stage2.csv'), header=None, names=['id', 'class_num'])
    transform = eval(config.transform.name)(config.transform.size)

    X_test = config.root + config.test_path + test_df['id']
    # X_test = test_df['id']
    test_data = TestDataset(X_test, transform['albu_val'], test_path)
    test_loader = DataLoader(test_data, **config.val_loader)
    states = []
    for i in range(len(model_name)):
        states.append(load_state(model_name[i], model_path[i]))

    model = eval(config.model)(True)
    if 'fc.weight' in model.state_dict().keys():
        model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
    elif 'classifier.weight' in model.state_dict().keys():
        model.classifier = nn.Linear(model.classifier.in_features, config.train.num_labels)
    elif 'head.fc.weight' in model.state_dict().keys():
    # model.load_state_dict(torch.load('./pretrained/NFNet-f1.pt'))
        model.head.fc = nn.Linear(model.head.fc.in_features, config.train.num_labels)

    predictions = inference(model, states, test_loader, 'cuda', transform['torch_val'])
    predict_df = pd.DataFrame()
    predict_df['id'] = test_df['id']
    predict_df['label'] = predictions.argmax(1)
    predict_df.to_csv('predict.csv', index=False, header=False)


if __name__ == '__main__':
    main()
