import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.reParams['figure.figsize'] = (12, 12)
# mpl.reParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    # define a function to load an image and limits its maximum dimension to 512 pixels
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    # 设置输入图片channel为3
    img = tf.image.decode_image(img, channels=3)
    # 数据类型转化为32位浮点数
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    # 使用scale进行同比例缩放
    scale = max_dim / long_dim

    # 这里操作的数据类型均为tf.Tensor()
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    # create a simple
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    # 使VGG输出指定layer的内容
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result/num_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, training=None, mask=None):
        inputs = inputs * 255
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


if __name__ == '__main__':
    # content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
    #                                        'https://storage.googleapis.com/download.tensorflow.org'
    #                                        '/example_images/YellowLabradorLooking_new.jpg')
    content_path = "C:/Users/Herrrb/Desktop/nrma.jpg"
    style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                         'https://storage.googleapis.com/download.tensorflow.org/'
                                         'example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    # print(content_path)
    # print(style_path)
    # exit(0)

    # plt.subplot(1, 2, 1)
    # imshow(content_image, "Content Image")
    # plt.subplot(1, 2, 2)
    # imshow(style_image, "Style Image")
    # plt.show()

    # define
    # x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    # x = tf.image.resize(x, (224, 224))
    # vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    # prediction_probabilities = vgg(x)
    # print(prediction_probabilities.shape)
    #
    # predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    # print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # style_extractor = vgg_layers(style_layers)
    # style_output = style_extractor(style_image * 255)
    #
    # for name, output in zip(style_layers, style_output):
    #     print(name)
    #     print("  shape:  ", output.numpy().shape)
    #     print("  min:  ", output.numpy().min())
    #     print("  max:  ", output.numpy().max())
    #     print("  mean:  ", output.numpy().mean())

    extractor = StyleContentModel(style_layers, content_layers)
    # results = extractor(tf.constant(content_image))
    #
    # print('Style:')
    # for name, output in sorted(results['style'].items()):
    #     print("    ", name)
    #     print("    shape:    ", output.numpy().shape)
    #     print("    min:    ", output.numpy().min())
    #     print("    max:    ", output.numpy().max())
    #     print("    mean:    ", output.numpy().mean())
    #
    # print('Content:')
    # for name, output in sorted(results['content'].items()):
    #     print("    ", name)
    #     print("    shape:    ", output.numpy().shape)
    #     print("    min:    ", output.numpy().min())
    #     print("    max:    ", output.numpy().max())
    #     print("    mean:    ", output.numpy().mean())

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        # reduce_mean计算张量tensor沿着指定的数轴上的平均值，主要用作降维或计算tensor的平均值
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name_] - style_targets[name_])**2)
                               for name_ in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name_] - content_targets[name_])**2)
                                 for name_ in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)

        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    train_step(image)
    train_step(image)
    train_step(image)
    img = tensor_to_image(image)
    img.show()

    start = time.time()
    epochs = 10
    steps_per_epoch = 100
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end=' ')
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    img = tensor_to_image(image)
    img.show()
