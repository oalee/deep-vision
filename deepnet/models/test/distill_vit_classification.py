from argparse import Namespace
from pytorch_lightning import LightningModule
import torch as t
import torchmetrics
import mate
import ipdb


class DistillClassificationModule(LightningModule):

    def __init__(self, params: Namespace, distiller: t.nn.Module, *args):
        super().__init__(*args)

        self.params = params
        self.save_hyperparameters(params)
        self.criterion = t.nn.CrossEntropyLoss()

        self.distiller = distiller

        # disable gradient tracking for teacher
        for p in self.distiller.teacher.parameters():
            p.requires_grad = False
        

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.loss = lambda y_hat, y: self.criterion(y_hat, y)

    def forward(self, x):
        return self.distiller.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.train_accuracy.update(y_hat, y)

        loss = self.distiller(x, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.distiller(x, y)
        self.val_accuracy.update(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        loss_mean = t.stack([x for x in outputs]).mean()
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_loss', loss_mean)
        self.val_accuracy.reset()

    def training_epoch_end(self, outputs):
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return mate.Optimizer.optimizers(self.params.optimizers, self)
