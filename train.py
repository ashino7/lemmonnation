import os
import atexit
import argparse
import warnings

import timm
import torch
import numpy as np
import torch.nn as nn
import mlcrate as mlc
from omegaconf import OmegaConf
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from timm.models import *
from sklearn.metrics import accuracy_score, cohen_kappa_score
from dataset import MyDataset, get_image_and_label
from augmentations.strong_aug import *
from augmentations.augmentation import *
from utils import find_exp_num, get_logger, remove_abnormal_exp, seed_everything, save_model, save_model_compact
from losses.loss import TaylorCrossEntropyLoss

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", default='./config/base.yaml')
    parser.add_argument("options", nargs="*")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(args.options)
    atexit.register(remove_abnormal_exp, log_path=config.log_path, config_path=config.config_path)  # 不要なexp削除用
    seed_everything(config.seed)  # seedを固定

    exp_num = find_exp_num(log_path=config.log_path)
    exp_num = str(exp_num).zfill(3)
    config.weight_path = os.path.join(config.weight_path, f'exp_{exp_num}')
    os.makedirs(config.weight_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.config_path, f'exp_{exp_num}.yaml'))
    logger, csv_logger = get_logger(config, exp_num)
    timer = mlc.time.Timer()
    logger.info(mlc.time.now())
    logger.info(f'config: {config}')

    transform = eval(config.transform.name)(config.transform.size)
    logger.info(f'augmentation: {transform}')
    strong_transform = eval(config.strong_transform.name)
    logger.info(f'strong augmentation: {config.strong_transform.name}')

    x_train, y_train = get_image_and_label(config.train_path)
    x_val, y_val = get_image_and_label(config.val_path)

    train_data = MyDataset(x_train, y_train, transform['albu_train'])
    val_data = MyDataset(x_val, y_val, transform['albu_val'])
    train_loader = DataLoader(train_data, **config.train_loader)
    val_loader = DataLoader(val_data, **config.val_loader)

    model = timm.create_model(config.model, pretrained=True, num_classes=config.train.num_labels)

    # model = eval(config.model)(True)
    # if 'fc.weight' in model.state_dict().keys():
    #     model.fc = nn.Linear(model.fc.in_features, config.train.num_labels)
    # elif 'classifier.weight' in model.state_dict().keys():
    #     model.classifier = nn.Linear(model.classifier.in_features, config.train.num_labels)
    # elif 'head.fc.weight' in model.state_dict().keys():
    #     # model.load_state_dict(torch.load('./pretrained/NFNet-f1.pt'))
    #     model.head.fc = nn.Linear(model.head.fc.in_features, config.train.num_labels)
    # else:
    #     model = timm.create_model(config.model, pretrained=True, num_classes=config.train.num_labels)

    model = model.cuda()
    optimizer = eval(config.optimizer.name)(model.parameters(), lr=config.optimizer.lr)
    scheduler = eval(config.scheduler.name)(optimizer, config.train.epoch // config.scheduler.cycle,
                                            eta_min=config.scheduler.eta_min)
    criterion = eval(config.loss)
    scaler = GradScaler()

    best_acc = 0
    best_loss = 1e10
    for epoch in range(config.train.epoch):
        # train start
        timer.add('train')
        train_loss, train_acc = train(
            config, model, transform['torch_train'], strong_transform, train_loader, optimizer, criterion, epoch, scaler)
        train_time = timer.fsince('train')

        # val start
        timer.add('val')
        val_loss, val_acc, preds, gt = validate(config, model, transform['torch_val'], val_loader, criterion)
        val_time = timer.fsince('val')

        output1 = f'epoch: {epoch} train_time: {train_time} validate_time: {val_time}'
        output2 = f'train_loss: {train_loss:.3f} train_acc: {train_acc:.3f} val_loss: {val_loss:.3f} val_acc: {val_acc:.3f}'
        logger.info(output1)
        logger.info(output2)
        print(output1)
        print(output2)
        csv_logger.write([epoch, train_loss, train_acc, val_loss, val_acc])

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            save_name = os.path.join(config.weight_path, f'best_loss.pth')
            # save_model(save_name, epoch, val_loss, val_acc, model, optimizer, preds, gt)
            save_model_compact(save_name, model)
        if val_acc > best_acc:
            best_acc = val_acc
            save_name = os.path.join(config.weight_path, f'best_acc.pth')
            # save_model(save_name, epoch, val_loss, val_acc, model, optimizer, preds, gt)
            save_model_compact(save_name, model)

        save_name = os.path.join(config.weight_path, f'last_epoch.pth')
        # save_model(save_name, epoch, val_loss, val_acc, model, optimizer, preds, gt)
        save_model_compact(save_name, model)
    output3 = f'Best Loss: {best_loss:.3f} Best Acc: {best_acc:.3f}'
    logger.info(output3)
    # check_point = torch.load(os.path.join(config.weight_path, f'fold{fold}_best_acc.pth'))
    c=0


@torch.enable_grad()
def train(config, model, transform, strong_transform, loader, optimizer, criterion, epoch, scaler):
    preds = []
    gt = []
    losses = []
    scores = []

    model.train()
    for it, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)
        if epoch < config.train.epoch - 5:  # 学習終了間際までstrong_transformをやる場合（学習終了間際までやっても別に良い）
            with autocast():
                images, labels_a, labels_b, lam = strong_transform(images, labels, **config.strong_transform.params)
                logits = model(images)
                loss = criterion(logits, labels_a) * lam + criterion(logits, labels_b) * (1 - lam)

        else:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

        # updater関係
        scaler.scale(loss).backward()
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

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = accuracy_score(gt, preds)
    return np.mean(losses), score


@torch.no_grad()
def validate(config, model, transform, loader, criterion):
    preds = []
    gt = []
    losses = []

    model.eval()
    for it, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()
        images = transform(images)

        logits = model(images)
        loss = criterion(logits, labels)

        logits = (logits.argmax(1)).cpu().numpy().astype(int)
        labels = labels.cpu().numpy().astype(int)
        preds.append(logits)
        gt.append(labels)
        losses.append(loss.item())

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    score = accuracy_score(gt, preds)
    return np.mean(losses), score, preds, gt


if __name__ == '__main__':
    main()
