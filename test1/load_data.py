import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib

from tensorflow.keras import layers

if __name__ == '__main__':
    data_dir = "C:/Users/Herrrb/Desktop/tf-examples/test1/flower_photos/"
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))

    batch_size = 32
    img_height = 180
    img_width = 180

    # 这里image_dataset_from_directory返回的是BatchDataset类型
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,

    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    # 这里take就变成了TakeDataset类型，不知道是什么数据类型
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            # you can call .numpy() on tensors to convert them to a numpy.ndarray
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.show()

    for image_batch, label_batch in train_ds:
        print(image_batch.shape)
        print(label_batch.shape)
        break

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    # 开始训练模型
    num_classes = 5
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_ds, epochs=3)
