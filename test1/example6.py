import IPython
import tensorflow as tf
import kerastuner as kt
from tensorflow import keras


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
    # 将值限制在0-1之间
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0

    # 实例化（Instantiate）tuner，使用hypertuning
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    tuner.search(img_train, label_train, epochs=10, validation_data=(img_test, label_test))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is "
          f"{best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")

    model = tuner.hypermodel.build(best_hps)
    model.fit(img_train, label_train, epochs=10, validation_data=(img_test, label_test))
    loss, acc = model.evaluate(img_test, label_test, verbose=2)
    print("Hyper Tuning result: {:5.2f}%".format(acc*100))
