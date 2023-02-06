from ...data.cifar10 import CifarLightningDataModule
from ...trainers.classification.pl_classification import LightningClassificationModule
from ...models.resnet import ResNetTuneModel
from torchvision.models import resnet18, resnet34, ResNet34_Weights
import sys
import os
import torch
import pytorch_lightning as pl


network = ResNetTuneModel(
    resnet34(weights=ResNet34_Weights.IMAGENET1K_V1), num_classes=10
)

optimizer = torch.optim.Adam(network.parameters(), lr=0.0004, betas=[0.9, 0.999])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=0,
    verbose=True,
    threshold=1e-06,
)


optimizer = {
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "monitor": "val_loss",
}

pl_module = LightningClassificationModule(network, optimizer)

data_path = os.environ.get("DATA_PATH", "./data")
data_module = CifarLightningDataModule(data_path, batch_size=64, image_size=[32, 32])

save_path = os.environ.get("SAVE_PATH", "./results")
chkpt_name = "cifar10_resnet_fine_tune"
save_ckpt = os.path.join(save_path, chkpt_name)

logger = pl.loggers.TensorBoardLogger("tb_logs", name="cifar10_resnet")
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
        dirpath=save_ckpt,
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

command = sys.argv[1]

if command == "train":
    pl_trainer.fit(pl_module, data_module)
if command == "restart":
    pl_trainer.fit(
        pl_module, data_module, ckpt_path=os.path.join(save_ckpt, "last.ckpt")
    )
elif command == "test":
    pl_trainer.test(pl_module, data_module, ckpt_path=save_ckpt)
