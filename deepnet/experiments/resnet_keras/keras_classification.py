from ...models.keras.resnet import get_model
import keras, sys, os


resnet: keras.Model = get_model(input_shape=(32, 32, 3), num_classes=10)

resnet.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

save_folder = os.environ.get("SAVE_PATH", "./results")
save_path = os.path.join(save_folder, "cifar10_resnet.h5")

command = sys.argv[1]
if command == "train":
    resnet.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        epochs=10,
    )
    resnet.save(save_path)
elif command == "test":
    resnet.load_weights(save_path)
    resnet.evaluate(x_test, y_test)
elif command == "restart":
    resnet.load_weights(save_path)
    resnet.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=64,
        epochs=10,
    )
    resnet.save(save_path)
