import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import random


def main_1():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)


def main_2():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("image的格式为：" + str(train_images.shape))

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)
    print("\nTest Loss:", test_loss)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print("---------------------------------------------------")

    print(np.argmax(predictions[0]))
    print("This image is an " + class_names[int(np.argmax(predictions[0]))])

    img = test_images[124]
    print(img.shape)
    img = (np.expand_dims(img, 0))
    print(img.shape)

    predictions_single = probability_model.predict(img)
    print(predictions_single)
    print("The result is " + class_names[int(np.argmax(predictions_single))])
    print("The true label is " + class_names[test_labels[124]])

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def build_data():
    num_sha1 = 0
    num_md5 = 0
    num_mysql = 0
    with open("jiami_v3.txt", "w") as f:
        with open("sha1.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_sha1 += 1
                    index = h.find(":")
                    h = h[:index+1] + "0\n"
                    # SHA1
                    f.write(h)
                else:
                    break
        with open("sha1_2.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_sha1 += 1
                    index = h.find(":")
                    h = h[:index+1] + "0\n"
                    # SHA1
                    f.write(h)
                else:
                    break
        with open("MD5_401.txt", "r") as ff:
            while 1:
                h = ff.readline().strip('\n')
                if h:
                    num_md5 += 1
                    h = h + ":1\n"
                    # MD5
                    f.write(h)
                else:
                    break
        with open("mysql5.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_mysql += 1
                    index = h.find(":")
                    h = h[:index+1] + "2\n"
                    # MYSQL
                    f.write(h)
                else:
                    break
        with open("mysql_2.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_mysql += 1
                    index = h.find(":")
                    h = h[:index+1] + "2\n"
                    f.write(h)
                else:
                    break
        with open("mysql_3.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_mysql += 1
                    index = h.find(":")
                    h = h[:index+1] + "2\n"
                    f.write(h)
                else:
                    break
        with open("mysql_4.txt", "r") as ff:
            while 1:
                h = ff.readline()
                if h:
                    num_mysql += 1
                    index = h.find(":")
                    h = h[:index+1] + "2\n"
                    f.write(h)
                else:
                    break
        with open("mysql_5.txt", "r") as ff:
            while 1:
                h = ff.readline().strip("*")
                if h:
                    num_mysql += 1
                    index = h.find(":")
                    h = h[:index+1] + "2\n"
                    f.write(h)
                else:
                    break
    print("SHA1数量：" + str(num_sha1))
    print("MD5数量：" + str(num_md5))
    print("MYSQL数量：" + str(num_mysql))

    data_s = []
    with open("jiami_v3.txt", "r") as f:
        while 1:
            h = f.readline().strip("\n")
            if h:
                data_s.append(h)
            else:
                break

    random.shuffle(data_s)

    i = len(data_s) * 0.3
    train = data_s[int(i):]
    test = data_s[:int(i)]

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in train:
        h = i.split(":")
        x = np.array([[ord(str(i))] for i in h[0]], dtype='int16')
        train_x.append(x)
        train_y.append(int(h[1]))

    for i in test:
        h = i.split(":")
        x = np.array([[ord(str(i))] for i in h[0]], dtype='int16')
        test_x.append(x)
        test_y.append(int(h[1]))

    train_x = np.asarray(train_x, dtype='int16')
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x, dtype='int16')
    test_y = np.asarray(test_y)

    print("处理完成")

    return (train_x, train_y), (test_x, test_y)


def main():
    (train_x, train_y), (test_x, test_y) = build_data()
    print(train_x.shape)
    class_names = ['SHA1', 'MD5', 'MYSQL']

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(40, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=3)

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
    print("\nTest accuracy:", test_acc)
    print("\nTest Loss:", test_loss)

    prediction_model = keras.models.Sequential([model, keras.layers.Softmax()])

    mi1 = "62ffe6ed9eb1f842e1247defbd27906b667a9182"  # SHA1
    mi2 = "cdc4fecad57588abdbdf4eaf7b15018992b0f652"  # MYSQL
    mi3 = "320c246daaae25c66081f6dbaaae25c66081f6db"  # MD5

    mi1 = np.array([[ord(i)] for i in mi1], dtype='int16')
    mi2 = np.array([[ord(i)] for i in mi2], dtype='int16')
    mi3 = np.array([[ord(i)] for i in mi3], dtype='int16')

    mi1 = np.expand_dims(mi1, 0)
    mi2 = np.expand_dims(mi2, 0)
    mi3 = np.expand_dims(mi3, 0)

    prediction1 = prediction_model.predict(mi1)
    prediction2 = prediction_model.predict(mi2)
    prediction3 = prediction_model.predict(mi3)

    print("One is " + class_names[int(np.argmax(prediction1))])
    print("Two is " + class_names[int(np.argmax(prediction2))])
    print("Three is " + class_names[int(np.argmax(prediction3))])


if __name__ == "__main__":
    main()
