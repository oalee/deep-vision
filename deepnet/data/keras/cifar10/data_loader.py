import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


class DataLoader:
    def __init__(self, batch_size: int = 32):
        if not os.path.exists("PetImages"):
            os.system(
                "curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
            )
            os.system("unzip -q kagglecatsanddogs_5340.zip")

            num_skipped = 0
            for folder_name in ("Cat", "Dog"):
                folder_path = os.path.join("PetImages", folder_name)
                for fname in os.listdir(folder_path):
                    fpath = os.path.join(folder_path, fname)
                    try:
                        fobj = open(fpath, "rb")
                        is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                    finally:
                        fobj.close()

                    if not is_jfif:
                        num_skipped += 1
                        # Delete corrupted image
                        os.remove(fpath)
        image_size = (180, 180)
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        augmented_train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )
        self.train_ds = augmented_train_ds.prefetch(buffer_size=32)
        self.val_ds = val_ds.prefetch(buffer_size=32)

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds

    def test_dataloader(self):
        return self.val_ds


loader = DataLoader()


def get_train_data_loader(batch_size: int = 32):

    return loader.train_ds


def get_data_loader(batch_size: int = 32):

    return loader.train_ds, loader.val_ds
