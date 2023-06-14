import tensorflow as tf
from tensorflow import keras

@profile
def train():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


    x_train = x_train / 255.0
    x_test = x_test / 255.0


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    history = model.fit(x_train, y_train, epochs=1, batch_size=32)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    del x_train, x_test, y_test, y_train
    del model
    del history
    del test_loss, test_acc
    import gc
    gc.collect()


train()
del train
import gc
gc.collect()

