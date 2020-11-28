import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


if __name__ == '__main__':
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    encoder = info.features['text'].encoder
    # print("Vocabulary size: {}".format(encoder.vocab_size))
    #
    # sample_string = "Hello Tensorflow."
    # encoded_string = encoder.encode(sample_string)
    # print("Encoded string is {}".format(encoded_string))
    #
    # for index in encoded_string:
    #     print("{} ----> {}".format(index, encoder.decode([index])))

    BUFFER_SIZE = 1000
    BATCH_SIZE = 32

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)
    test_dataset = test_dataset.padded_batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 32),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset,
                        validation_steps=30)

    test_loss, test_acc = model.evaluate(test_dataset)
    print("Test loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_acc))
