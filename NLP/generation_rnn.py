import tensorflow as tf

import numpy as np
import os
import time


def build_model(vocab_size_, embedding_dim, rnn_units_, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size_, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units_,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        # Glorot均匀分布初始化器，也称为Xavier均匀分布初始化器
        # 从[-limit, limit]中的均匀分布中抽取样本
        # limit是sqrt(6 / (权值张量输入单位的数量 + 输出单位的数量))
        tf.keras.layers.Dense(vocab_size_)
    ])
    return model


if __name__ == '__main__':
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    text = open("./shakespeare.txt", 'rb').read().decode("ascii")
    print(f"全文共{len(text)}个字符")
    print(text[:250])

    vocab = sorted(set(text))
    print(f"{len(vocab)}个独立的字符")

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.asarray(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    print("{")
    for char_, _ in zip(char2idx, range(20)):
        print(" {:4s}: {:3d},".format(repr(char_), char2idx[char_]))
    print(" ...\n}")

    print("{} ---- characters mapped to int ---- > {}".format(repr(text[:13]), text_as_int[:13]))

    seq_length = 100
    example_per_epoch = len(text) // (seq_length+1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequence = char_dataset.batch(seq_length+1, drop_remainder=True)

    for item in sequence.take(5):
        print(repr(''.join(idx2char[item.numpy()])))
        print("\n")

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequence.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data:', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(vocab)
    embedding_size = 256
    rnn_units = 1024

    # model = build_model(vocab_size_=vocab_size, embedding_dim=embedding_size,
    #                     rnn_units_=rnn_units, batch_size=BATCH_SIZE)
    #
    # def loss(labels, logits):
    #     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    #
    # model.compile(optimizer='adam', loss=loss)
    #
    checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    #
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_prefix,
    #     save_weights_only=True
    # )
    #
    # EPOCHS = 10
    # history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    model = build_model(vocab_size_=vocab_size, embedding_dim=embedding_size,
                        rnn_units_=rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))
    print(model.summary())

    def generate_text(model_, start_string):
        num_generate = 1000
        input_eval = [char2idx[s] for s in start_string]
        # 问题：expand_dims的作用
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        # low temperature可以产生更多的text
        # higher temperature可以产生更令人惊奇的text
        # 具体数字是多少需要我们慢慢实验
        temperature = 1.0

        model_.reset_states()
        for i in range(num_generate):
            predictions = model_(input_eval)
            # squeeze的作用
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return start_string + ''.join(text_generated)

    print(generate_text(model, start_string=u"REMEO: "))

    # model = build_model(vocab_size, embedding_size, rnn_units, BATCH_SIZE)
    # optimizer = tf.keras.optimizers.Adam()
    #
    # @tf.function
    # def train_step(inp, target):
    #     with tf.GradientTape() as tape:
    #         predictions = model(inp)
    #         loss = tf.reduce_mean(
    #             tf.keras.losses.sparse_categorical_crossentropy(
    #                 target, predictions, from_logits=True
    #             )
    #         )
    #     grads = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     return loss

    # EPOCHS = 10
    # for epoch in range(EPOCHS):
    #     start = time.time()
    #     model.reset_states()
    #     for (batch_n, (inp, target)) in enumerate(dataset):
    #         loss = train_step(inp, target)
    #         if batch_n % 100 == 0:
    #             template = "Epoch {} Batch {} Loss {}"
    #             print(template.format(epoch+1, batch_n, loss))
    #     if (epoch + 1) % 5 == 0:
    #         model.save_weights("./herbwen/ckpt-{}".format(epoch))
    #
    #     print("Epoch {} Loss {:.4f}".format(epoch+1, loss))
    #     print("Time taken for 1 epoch {} sec\n".format(time.time() - start))
    # model.save_weights("./herbwen/ckpt-{}".format(epoch))
