from tensorflow import keras
from keras.applications.resnet import ResNet50
import ipdb 

resnet: keras.Model = ResNet50(
    include_top=False,
    input_tensor=keras.Input(shape=(32, 32, 3)),
    classes=10,
    classifier_activation="softmax",
)



resnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.9, beta_2=0.999),
    loss="binary_crossentropy",
    metrics=["accuracy", "loss"],
)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

resnet.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=10,
)
