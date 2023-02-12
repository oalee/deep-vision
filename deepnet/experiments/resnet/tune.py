from ...data.cifar10 import CifarLightningDataModule
from ...trainers.classification import LightningClassificationModule
from ...models.resnet import ResNetTuneModel
from torchvision.models import resnet18, resnet34
from timm.data.auto_augment import rand_augment_transform
import torchvision
import sys, os, torch, pytorch_lightning as pl, yerbamate, tensorboard
from torchvision import transforms

env = yerbamate.Environment()
network = ResNetTuneModel(
    resnet34(pretrained=True), num_classes=10, update_all_params=True
)
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0004, betas=[0.9, 0.999])
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    verbose=True,
    threshold=1e-06,
)
optimizer = {
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "monitor": "val_loss",
}

image_size = [32, 32]

train_transform = torchvision.transforms.Compose(
    [
        rand_augment_transform("rand-m9-n4-mstd0.5", {}),
        transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

pl_module = LightningClassificationModule(network, optimizer)
data_module = CifarLightningDataModule(
    env["data"], batch_size=128, image_size=[32, 32], train_transform=train_transform, test_transform=test_transform
)
save_path = env["results"]
logger = pl.loggers.TensorBoardLogger(env["logs"], env.name)

callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
        dirpath=save_path,
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
        pl_module, data_module, ckpt_path=os.path.join(save_path, "last.ckpt")
    )
elif env.test:
    pl_trainer.test(pl_module, data_module, ckpt_path=save_path)
