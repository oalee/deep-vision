from ...data.cifar10 import CifarLightningDataModule
from ...trainers.classification import LightningClassificationModule
from deepnet.models.resnet import ResNetTuneModel
from torchvision.models import resnet18, resnet34
import sys, os, torch, pytorch_lightning as pl, yerbamate 

env = yerbamate.Environment()
network = ResNetTuneModel(
    resnet34(pretrained=True), num_classes=10, update_all_params=True
)
optimizer = torch.optim.Adam(network.parameters(), lr=0.0004, betas=[0.9, 0.999])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    verbose=True,
    threshold=1e-06,
)
optimizer = {
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "monitor": "val_loss",
}

pl_module = LightningClassificationModule(network, optimizer)
data_module = CifarLightningDataModule(env["DATA_PATH"], batch_size=128, image_size=[32, 32])
logger = pl.loggers.TensorBoardLogger(env["logs"], env.name) 
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
        dirpath= env["results"],
        save_last=True,
    ),
    pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
]
pl_trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=100,
    logger=logger,
    callbacks=callbacks,
    precision=16,
    gradient_clip_val=0.5,
)
if env.train:
    pl_trainer.fit(pl_module, data_module)
if env.restart:
    pl_trainer.fit(
        pl_module, data_module, ckpt_path=os.path.join( env["results"], "last.ckpt")
    )
elif env.test: 
    pl_trainer.test(pl_module, data_module, ckpt_path=os.path.join( env["results"], "last.ckpt"))
