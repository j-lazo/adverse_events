from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import applications


input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 224), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299),
                      'resnet152': (224, 224), 'densenet201': (224, 224)}


def get_preprocess_input_backbone(name_backbone, x):
    """

    :param name_backbone:
    :param x:
    :return:
    """
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


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False):
    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """

    if name_model == 'vgg16':
        base_model = applications.vgg16.VGG16(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'vgg19':
        base_model = applications.vgg19.VGG19(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'inception_v3':
        base_model = applications.inception_v3.InceptionV3(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'resnet50':
        base_model = applications.resnet50.ResNet50(include_top=include_top, weights=weights)
        base_model.trainable = True

    elif name_model == 'resnet101':
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights)
        base_model.trainable = True

    elif name_model == 'mobilenet':
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'densenet121':
        base_model = applications.densenet.DenseNet121(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'xception':
        base_model = applications.xception.Xception(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'resnet152':
        base_model = applications.resnet.ResNet152(include_top=include_top, weights=weights)
        base_model.trainable = trainable

    elif name_model == 'densenet201':
        base_model = applications.densenet.DenseNet201(include_top=include_top, weights=weights)
        base_model.trainable = trainable


    else:
        raise ValueError(f' MODEL: {name_model} not found')

    new_base_model = tf.keras.models.clone_model(base_model)
    new_base_model.set_weights(base_model.get_weights())

    return new_base_model


def simple_classifier(num_classes, backbone='resnet101', input_size=input_sizes_models['resnet101']):
    input_image = keras.Input(shape=input_size + (3,), name="image_input")

    x = tf.image.resize(input_image, input_sizes_models[backbone], method='area')
    x = get_preprocess_input_backbone(backbone, x)
    base_model = load_pretrained_backbones(backbone)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=input_image, outputs=output_layer, name=f'pretrained_model_{backbone}')
