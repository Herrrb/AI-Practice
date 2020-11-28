import string
import numpy as np
import tensorflow as tf


def split_input_target(chunk):
    input_ = chunk[:-1]
    target_ = chunk[1:]
    return input_, target_


def build_model(word_size, embedding_dim, rnn_unit, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_unit,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(word_size)
    ])
    return model


def losses(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


if __name__ == '__main__':
    text = open("./shakespeare.txt", "rb").read().decode("ascii").replace('--', ' ').replace("'", ' ')
    text_ = text.replace('--', ' ')
    punc_token = ['\n', ' ', '!', '$', '&', ',', '.', '3', ':', ';', '?']
    for i in string.punctuation:
        if i == '-':
            continue
        text = text.replace(i, ' ')
    text = text.replace('\n', ' ')
    text_list = text.split(' ') + punc_token
    text_set = sorted(set(text_list))

    word2idx = {i: u for u, i in enumerate(text_set)}
    idx2word = np.asarray(text_set)

    word_as_int = []

    word = ''
    for i in text_:
        if i in punc_token:
            if word != '':
                word_as_int.append(word2idx[word])
                word = ''
            word_as_int.append(word2idx[i])
        else:
            word += i
    # print(text_[:500])
    # for i in word_as_int:
    #     print(idx2word[i], end='')
    # 接下来一步是划分batch
    seq_length = 100
    word_dataset = tf.data.Dataset.from_tensor_slices(word_as_int)
    dataset = word_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = dataset.map(split_input_target)

    EPOCHS = 20
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    word_size = len(text_set)
    embedding_dim = 256
    rnn_units = 1024
    # model = build_model(word_size, embedding_dim, rnn_units, BATCH_SIZE)
    # model.compile(optimizer='Adam', loss=losses)
    checkpoints_dir = './herrrb-word-version'
    # checkpoints_prefix = checkpoints_dir + "/ckpt-{epoch}"
    # checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoints_prefix,
    #     save_weights_only=True
    # )
    # model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoints_callback])

    model = build_model(word_size, embedding_dim, rnn_units, 1)
    model.load_weights(tf.train.latest_checkpoint(checkpoints_dir))
    model.build(tf.TensorShape([1, None]))

    def generate_text(model_, start_string):
        num_generate = 1000
        input_eval = [word2idx[i] for i in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        generate = start_string
        model.reset_states()
        for i in range(num_generate):
            predictions = model_(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            if predictions_id == 2:
                print("2")

            input_eval = tf.expand_dims([predictions_id], 0)
            generate.append(idx2word[predictions_id])
        return generate

    generate = generate_text(model, ["All", ":", "\n", "\n"])
    for i in generate:
        if ord(i[0]) < ord("Z"):
            print("\n" + i)
        print(i, end=" ")
