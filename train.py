import argparse
import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex

from dataset import MyDataset
from deeplabv3 import DeepLabv3Plus
from utils import compute_supervised_loss
from stream import ProgressStream, CustomError


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class GradualWarmupScheduler(_LRScheduler):
    # https://github.com/seominseok0429/pytorch-warmup-cosine-lr
    def __init__(self, optimizer, multiplier, total_warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_warmup_epoch = total_warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * (
                (self.multiplier - 1.0) *
                self.last_epoch / self.total_warmup_epoch
                + 1.0
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_warmup_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def cosine_warmup(optimizer, total_epoch, num_warmup_epoch):
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_epoch, eta_min=0, last_epoch=-1
    )
    return GradualWarmupScheduler(
        optimizer,
        multiplier=8,
        total_warmup_epoch=num_warmup_epoch,
        after_scheduler=cosine_scheduler,
    )


def get_transform(mode='train'):
    if mode == 'train':
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.HueSaturationValue(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Cutout(num_holes=10, max_h_size=16, max_w_size=16,
                         fill_value=0, always_apply=False, p=0.5),
                A.Resize(384, 384),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )
    elif mode == "test":
        return A.Compose(
            [
                A.Resize(384, 384),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ]
        )


def dataloader(args, color_dict={}):
    train_set = MyDataset(
        root_dir=args.train_dir,
        color_dict=color_dict,
        transform=get_transform('train'))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.worker)

    test_set = MyDataset(
        root_dir=args.test_dir,
        color_dict=color_dict,
        transform=get_transform('test'))
    test_loader = DataLoader(
        test_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.worker)

    return train_loader, test_loader


def train(args):
    try:
        with open(args.classes_path) as json_file:
            color_dict = json.load(json_file)
        num_classes = len(color_dict)

        train_loader, test_loader = dataloader(args, color_dict=color_dict)

        if args.gpu:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')

        model = DeepLabv3Plus(models.resnet50(
            pretrained=True), num_classes=num_classes+1)
        # if args.finetune_model:
        #     model.load_state_dict(torch.load(model_path))
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = cosine_warmup(optimizer, args.epochs,  2)

        # Set up metrics
        if num_classes == 1:
            metrics = JaccardIndex(task='binary', num_classes=num_classes)
        else:
            metrics = JaccardIndex(task='multiclass', num_classes=num_classes)

        metrics.to(device)
        best_iou = -1.0
        print('Start Training')
        for epoch in tqdm(range(args.epochs),
                          file=ProgressStream(args.task_id)):
            total_loss = []
            print("Epoch {}/{}".format(epoch+1, args.epochs))

            for data in tqdm(train_loader):
                images, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                preds = model(images)
                preds = F.interpolate(preds, size=(
                    384, 384), mode='bilinear', align_corners=False)

                loss = compute_supervised_loss(preds, labels)
                total_loss.append(loss.item())
                loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()

            print('Epoch: %d, Loss: %.3f' % (epoch+1, np.mean(total_loss)))

            print("Eval")
            model.eval()

            iou = []
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    preds = model(images)
                    preds = F.interpolate(preds, size=(
                        384, 384), mode='bilinear', align_corners=True)
                    preds = torch.softmax(preds, dim=1)
                    preds = torch.argmax(preds, dim=1)
                    score = metrics(preds, labels)
                    iou.append(score.item())

            miou = np.mean(iou)
            print('Epoch: %d, mIoU: %.3f' % (epoch+1, miou))
            if miou > best_iou:
                model_path = os.path.join(
                    args.checkpoint_path,
                    'model_epoch_{}_acc_{}.pth'.format(
                        epoch+1,
                        round(miou, 2)))
                torch.save(model.state_dict(), model_path)
                best_iou = miou
                print('Saved new checkpoint')
                with open('result.json', 'w') as json_file:
                    json.dump({'iou': best_iou}, json_file)
            model.train()
            scheduler.step()
    except Exception as e:
        print(e)
        custom_err = CustomError(args.task_id,
                                 error_code="T02",
                                 message="Error while training crop model")
        custom_err.report()

        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--classes_path', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints/')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--worker', type=int, default=4)
    # parser.add_argument('--finetune_model', type=str, default=None)
    parser.add_argument('--task_id', type=str, default='',
                        help="Task's ID to update progress")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    train(args=args)
