from tensorflow import keras
from keras.applications import ResNet50

resnet: keras.Model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_tensor=keras.Input(shape=(32, 32, 3)),
    classes=10,
    classifier_activation="softmax",
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-6, decay_steps=10000, decay_rate=0.9
)

resnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

resnet.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=10,
)
