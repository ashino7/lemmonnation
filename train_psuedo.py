import os
import pprint
import atexit
from argparse import ArgumentParser
import random

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

from dataset import MetDataset, TestDataset
# from losses import lovasz_hinge
from augmentations.strong_aug import *
from augmentations.augmentation import met_transform1
from utils import find_exp_num, get_logger, remove_abnormal_exp, seed_everything, save_model
from tqdm import tqdm
from losses.loss import TaylorCrossEntropyLoss
import warnings
warnings.filterwarnings("ignore")


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-config", default='./config/base_psuedo.yaml')
    parser.add_argument("options", nargs="*")
    # args = parser.parse_args(args=['-config','config/base.yaml','options',''])
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(remove_abnormal_exp, log_path=config.log_path,
                    config_path=config.config_path)
    seed_everything(config.seed)

    exp_num = find_exp_num(log_path=config.log_path)
    exp_num = str(exp_num).zfill(3)
    config.weight_path = os.path.join(config.weight_path, f'exp_{exp_num}')
    os.makedirs(config.weight_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(
        config.config_path, f'exp_{exp_num}.yaml'))
    logger, csv_logger = get_logger(config, exp_num)
    timer = mlc.time.Timer()
    logger.info(mlc.time.now())
    logger.info(f'config: {config}')

    train_df = pd.read_csv(os.path.join(config.root, 'input/train_images_group_v2.csv'))
    test_df = pd.read_csv(os.path.join(config.root, 'input/sample_submit.csv'), header=None, names=['id', 'class_num'])
    X = train_df.id.values
    X = np.array([os.path.join(config.root, config.train_path, f'{i}') for i in X])
    y = train_df.class_num.values
    groups = train_df.group.values
    # y = np.load(os.path.join(config.root, 'labels.npy'))
    print(X.shape, y.shape)

    transform = eval(config.transform.name)(config.transform.size)
    logger.info(f'augmentation: {transform}')
    strong_transform = eval(config.strong_transform.name)
    logger.info(f'strong augmentation: {config.strong_transform.name}')

    folds = train_df.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['class_num'])):
        folds.loc[val_index, 'fold'] = int(n)
    # for n, (train_index, val_index) in enumerate(stratified_group_k_fold(X, y, groups, k=5, seed=config.seed)):
    #     folds.loc[val_index, 'fold'] = int(n)
    # folds['fold'] = folds['fold'].astype(int)

    for fold in range(config.train.n_splits):
        if fold not in config.train.trn_fold:
            continue
        train_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_data = MetDataset(X_train, y_train, transform['albu_train'])
        val_data = MetDataset(X_val, y_val, transform['albu_val'])
        train_loader = DataLoader(train_data, **config.train_loader)
        val_loader = DataLoader(val_data, **config.val_loader)

        transform = eval(config.transform.name)(config.transform.size)

        X_test = test_df['id']
        test_data = TestDataset(X_test, transform['albu_val'], config.test_path)
        test_loader = DataLoader(test_data, **config.val_loader)

        model = eval(config.model)(True)
        if 'fc.weight' in model.state_dict().keys():
            model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
        elif 'classifier.weight' in model.state_dict().keys():
            model.classifier = nn.Linear(model.classifier.in_features, config.train.num_labels)
        elif 'head.fc.weight' in model.state_dict().keys():
            # model.load_state_dict(torch.load('./pretrained/NFNet-f1.pt'))
            model.head.fc = nn.Linear(model.head.fc.in_features, config.train.num_labels)
        model = model.cuda()
        optimizer = eval(config.optimizer.name)(
            model.parameters(), lr=config.optimizer.lr)
        scheduler = eval(config.scheduler.name)(
            optimizer, config.train.epoch // config.scheduler.cycle, eta_min=config.scheduler.eta_min)
        criterion = eval(config.loss)(4)
        scaler = GradScaler()

        best_acc = 0
        best_loss = 1e10
        mb = master_bar(range(config.train.epoch))
        for epoch in mb:
            timer.add('train')
            train_loss, train_acc = train(
                config, model, transform['torch_train'], strong_transform, train_loader, optimizer, criterion, mb, epoch, scaler)
            train_time = timer.fsince('train')

            timer.add('val')
            val_loss, val_acc, preds, gt = validate(config, model, transform['torch_val'], val_loader, criterion, mb, epoch)
            val_time = timer.fsince('val')

            output1 = 'epoch: {} train_time: {} validate_time: {}'.format(
                epoch, train_time, val_time)
            output2 = 'train_loss: {:.3f} train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f}'.format(
                train_loss, train_acc, val_loss, val_acc)
            logger.info(output1)
            logger.info(output2)
            mb.write(output1)
            mb.write(output2)
            csv_logger.write(
                [epoch, train_loss, train_acc, val_loss, val_acc])

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = os.path.join(config.weight_path, f'fold{fold}_best_loss.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer, preds, gt)
            if val_acc > best_acc:
                best_acc = val_acc
                save_name = os.path.join(config.weight_path, f'fold{fold}_best_acc.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer, preds, gt)

            save_name = os.path.join(config.weight_path, f'fold{fold}_last_epoch.pth')
            save_model(save_name, epoch, val_loss,
                       val_acc, model, optimizer, preds, gt)

        # 疑似ラベルのためにtestを予測
        predictions = predict(config, model, transform['torch_val'], test_loader)
        test_df['class_num'] = predictions
        test_df['id'] = config.root + config.test_path + test_df['id']
        X_train = np.append(X_train, test_df.id.values)
        y_train = np.append(y_train, test_df.class_num.values)
        train_data = MetDataset(X_train, y_train, transform['albu_train'])
        train_loader = DataLoader(train_data, **config.train_loader)

        best_acc = 0
        best_loss = 1e10
        for epoch in mb:
            timer.add('train')
            train_loss, train_acc = train(
                config, model, transform['torch_train'], strong_transform, train_loader, optimizer, criterion, mb, epoch, scaler,pusedo=True)
            train_time = timer.fsince('train')

            timer.add('val')
            val_loss, val_acc, preds, gt = validate(config, model, transform['torch_val'], val_loader, criterion, mb, epoch)
            val_time = timer.fsince('val')

            output1 = 'epoch: {} train_time: {} validate_time: {}'.format(
                epoch, train_time, val_time)
            output2 = 'train_loss: {:.3f} train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f}'.format(
                train_loss, train_acc, val_loss, val_acc)
            logger.info(output1)
            logger.info(output2)
            mb.write(output1)
            mb.write(output2)
            csv_logger.write(
                [epoch, train_loss, train_acc, val_loss, val_acc])

            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                save_name = os.path.join(config.weight_path, f'fold{fold}_best_loss.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer, preds, gt)
            if val_acc > best_acc:
                best_acc = val_acc
                save_name = os.path.join(config.weight_path, f'fold{fold}_best_acc.pth')
                save_model(save_name, epoch, val_loss,
                           val_acc, model, optimizer, preds, gt)

            save_name = os.path.join(config.weight_path, f'fold{fold}_last_epoch.pth')
            save_model(save_name, epoch, val_loss,
                       val_acc, model, optimizer, preds, gt)

        output3 = 'Fold: {} Best Loss: {:.3f} Best Acc: {:.3f}'.format(fold, best_loss, best_acc)
        mb.write(output3)
        logger.info(output3)
        # check_point = torch.load(os.path.join(config.weight_path, f'fold{fold}_best_acc.pth'))


@torch.enable_grad()
def train(config, model, transform, strong_transform, loader, optimizer, criterion, mb, epoch, scaler, pusedo=False):
    preds = []
    gt = []
    losses = []
    scores = []

    model.train()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)
        if epoch < config.train.epoch - 5 or pusedo:
            images, labels_a, labels_b, lam = strong_transform(images, labels, **config.strong_transform.params)
            logits = model(images)
            loss = criterion(logits, labels_a) * lam + criterion(logits, labels_b) * (1 - lam)
            loss /= config.train.accumulate
            # with autocast():
            #     logits = model(images)
            #     loss = criterion(logits, labels_a) * lam + \
            #         criterion(logits, labels_b) * (1 - lam)
            #     loss /= config.train.accumulate
        else:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
                loss /= config.train.accumulate

        scaler.scale(loss).backward()
        if not (it + 1) % config.train.accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        logits = (logits.argmax(1)).detach().cpu().numpy().astype(int)
        labels = labels.detach().cpu().numpy().astype(int)
        score = accuracy_score(labels, logits)
        scores.append(score)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

        mb.child.comment = 'loss: {:.3f} avg_loss: {:.3f} acc: {:.3f} avg_acc: {:.3f}'.format(
            loss.item(),
            np.mean(losses),
            score,
            np.mean(scores),
        )

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = accuracy_score(gt, preds)
    return np.mean(losses), score


@torch.no_grad()
def validate(config, model, transform, loader, criterion, mb, device):
    preds = []
    gt = []
    losses = []

    model.eval()
    for it, (images, labels) in enumerate(progress_bar(loader, parent=mb)):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)

        logits = model(images)
        loss = criterion(logits, labels) / config.train.accumulate

        logits = (logits.argmax(1)).cpu().numpy().astype(int)
        labels = labels.cpu().numpy().astype(int)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = accuracy_score(gt, preds)
    return np.mean(losses), score, preds, gt


@torch.no_grad()
def predict(config, model, transform, loader):
    preds = []

    model.eval()
    for it, images in enumerate(loader):
        images = images.cuda()
        images = transform(images)

        logits = model(images)
        logits = (logits.argmax(1)).cpu().numpy().astype(int)
        preds.append(logits)

    preds = np.concatenate(preds)

    return preds


if __name__ == '__main__':
    main()
