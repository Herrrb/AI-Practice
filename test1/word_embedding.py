import io
import os
import re
import shutil
import string
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def custom_standardization(input_data):
    # 先定义一个过滤函数，将标点符号和HTML特殊字符过滤掉
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', " ")
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


if __name__ == '__main__':
    dataset_path = "C:\\Users\\Herrrb\\Desktop\\tf-examples\\test1\\aclImdb_v1\\aclImdb"
    train_dir = os.path.join(dataset_path, 'train')

    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)

    batch_size = 1024
    seed = 123
    train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb_v1/aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed
    )
    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb_v1/aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed
    )

    # 这里取的是第一个batch
    for text_batch, label_batch in train_ds.take(1):
        print(text_batch)
        print(label_batch)
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Embed a 1,000 word vocabulary into 5 dimensions.
    embedding_layer = tf.keras.layers.Embedding(1000, 5)

    # Vocabulary 大小，在sequence中的单词数量
    vocab_size = 100000
    sequence_length = 100

    # 使用text vectorization layer来标准化、分割、映射
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # 只取数据
    text_ds = train_ds.map(lambda x, y: x)
    # 训练这个向量化层
    vectorize_layer.adapt(text_ds)

    embedding_dim = 16
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=15,
              callbacks=[tensorboard_callback])

    print(model.summary())
