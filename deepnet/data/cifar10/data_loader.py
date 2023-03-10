# Cifar 10 data loader lightning class based on torchvision cifar10 dataset

from argparse import Namespace
from array import array
import os
import torch as t
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import ipdb

from torch.utils.data import DataLoader


class CifarLightningDataModule(pl.LightningDataModule):

    def __init__(self, location: str, batch_size: int, image_size: array, train_transform = None, test_transform=None ,*args):
        super().__init__(*args)

        self.data_dir = location
        self.batch_size = batch_size
        self.image_size = tuple(image_size)

        # transform pil image to tensor
        if train_transform is None:
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])
                ]
            )
        if test_transform is None:
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(self.image_size),
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])
                ]
            )

        self.train_set = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform)
        self.test_set = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform)

        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
