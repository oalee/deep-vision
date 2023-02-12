from ...data.cifar10 import CifarLightningDataModule
from ...trainers.classification import LightningClassificationModule
from ...models.resnet import ResNet, BasicBlock, Bottleneck
from ....timm.models import densenet, DenseNet
import sys
import os
import torch
import pytorch_lightning as pl


network = densenet.densenet121(pretrained=True, num_classes=10)
optimizer = torch.optim.Adam(network.parameters(), lr=0.0004, betas=(0.5, 0.99))    

pl_module = LightningClassificationModule(network, optimizer)

data_path = os.environ.get("DATA_PATH", "./data")
data_module = CifarLightningDataModule(data_path, batch_size=32, image_size=[32, 32])

save_path = os.environ.get("SAVE_PATH", "./results")
chkpt_name = "cifar10_resnet"
save_ckpt = os.path.join(save_path, chkpt_name)

logger = pl.loggers.TensorBoardLogger("tb_logs", name="cifar10_resnet")
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_accuracy", save_top_k=1, mode="max", dirpath=save_ckpt
    ),
    pl.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, mode="min"),
]

pl_trainer = pl.Trainer(
    gpus=1,
    max_epochs=100,
    logger=logger,
    callbacks=callbacks,
)

command = sys.argv[1]

if command == "train":
    pl_trainer.fit(pl_module, data_module)

elif command == "test":
    pl_trainer.test(pl_module, data_module, ckpt_path=save_ckpt)
