import keras
import tensorflow as tf
import os
from tensorflow.keras import applications

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 224), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299),
                          'resnet152': (224, 224), 'densenet201': (224, 224)}


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False, new_name=None):

    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """


    base_dir_weights = ''.join([os.getcwd(), '/scripts/models/weights_pretrained_backbones/'])
    if name_model == 'vgg16':
        weights_dir = base_dir_weights + 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg16.VGG16(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'vgg19':
        weights_dir = base_dir_weights + 'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg19.VGG19(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'inception_v3':
        weights_dir = base_dir_weights + 'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.inception_v3.InceptionV3(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'resnet50':
        weights_dir = base_dir_weights + 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet50.ResNet50(include_top=include_top, weights=weights_dir)
        base_model.trainable = True

    elif name_model == 'resnet101':
        weights_dir = base_dir_weights + 'resnet101/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights_dir)
        base_model.trainable = True

    elif name_model == 'mobilenet':
        weights_dir = base_dir_weights + 'mobilenet/mobilenet_1_0_224_tf_no_top.h5'
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'densenet121':
        weights_dir = base_dir_weights + 'densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.densenet.DenseNet121(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'xception':
        weights_dir = base_dir_weights + 'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.xception.Xception(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'resnet152':
        weights_dir = base_dir_weights + 'resnet152/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet152(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'densenet201':
        weights_dir = base_dir_weights + 'desenet201/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.densenet.DenseNet201(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    else:
        raise ValueError(f' MODEL: {name_model} not found')

    new_base_model = tf.keras.models.clone_model(base_model)
    new_base_model.set_weights(base_model.get_weights())

    return new_base_model


def get_preprocess_input_backbone(name_backbone, x):

    if name_backbone == 'resnet101':
        preprocess_input = tf.keras.applications.resnet.preprocess_input(x)
    elif name_backbone == 'resnet50':
        preprocess_input = tf.keras.applications.resnet50.preprocess_input(x)
    elif name_backbone == 'densenet121':
        preprocess_input = tf.keras.applications.densenet.preprocess_input(x)
    elif name_backbone == 'vgg19':
        preprocess_input = tf.keras.applications.vgg19.preprocess_input(x)
    elif name_backbone == 'vgg16':
        preprocess_input = tf.keras.applications.vgg19.preprocess_input(x)
    elif name_backbone == 'mobilenet':
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input(x)
    elif name_backbone == 'xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input(x)
    elif name_backbone == 'inception_v3':
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input(x)
    elif name_backbone == 'resnet152':
        preprocess_input = tf.keras.applications.resnet.preprocess_input(x)
    elif name_backbone == 'densenet201':
        preprocess_input = tf.keras.applications.densenet.preprocess_input(x)
    else:
        raise ValueError(f'Preprocess input {name_backbone} not avialable')
    return preprocess_input


def build_base_model(name_model, input_size=[256, 256, 3]):

    input_image = keras.Input(shape=input_size, name="image")
    x = tf.image.resize(input_image, input_sizes_models[name_model], method='area')
    x = get_preprocess_input_backbone(name_model, x)
    base_model = load_pretrained_backbones(name_model)
    for layer in base_model.layers:
        layer.trainable = False

    output_layer = base_model(x)

    return tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'pretrained_model_{name_model}')


def build_simple_rcnn(num_classes, backbone_network='efficient_net', optimizer='adam'):

    if backbone_network == 'efficient_net':
        net = tf.keras.applications.EfficientNetB0(include_top=False)
        #net = tf.keras.applications.ResNet50(include_top=False)
        net.trainable = False
    else:
        net = build_base_model(backbone_network)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=255),
        tf.keras.layers.TimeDistributed(net),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.GlobalAveragePooling3D()
    ])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def conv_3d(nb_classes, input_shape=(20, 256, 256, 3)):
    """
    Build a 3D convolutional network, based loosely on C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    """
    # Model.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(tf.keras.layers.Conv3D(64, (3,3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu'))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(tf.keras.layers.Conv3D(256, (2,2,2), activation='relu'))
    model.add(tf.keras.layers.Conv3D(256, (2,2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


def lstm(self):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model """
    # Model.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(2048, return_sequences=False,
                                   input_shape=self.input_shape,dropout=0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(self.nb_classes, activation='softmax'))

    return model


def build_pretrained_model(num_classes, name_model, input_size=[256, 256, 3]):
    input_image = keras.Input(shape=input_size, name="image")

    x = tf.image.resize(input_image, input_sizes_models[name_model], method='area')
    x = get_preprocess_input_backbone(name_model, x)
    base_model = load_pretrained_backbones(name_model)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'pretrained_model_{name_model}')