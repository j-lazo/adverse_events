import os
import numpy as np
import tensorflow as tf
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import tqdm
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow import keras
import time
from tensorflow.keras import applications


def load_model(directory_model):

    if directory_model == 'resnet50':
        model = applications.resnet50.ResNet50(include_top=True, weights='imagenet')
        input_size = None
    else:
        model_path = None
        if directory_model.endswith('.h5'):
            model_path = directory_model
        else:
            files_dir = [f for f in os.listdir(directory_model) if f.endswith('.h5')]
            if files_dir:
                model_path = files_dir.pop()
            else:
                files_dir = [f for f in os.listdir(directory_model) if f.endswith('.pb')]
                if files_dir:
                    model_path = ''
                    print(f'Tensorflow model found at {directory_model}')
                else:
                    print(f'No model found in {directory_model}')

        print('MODEL USED:')
        print(model_path)
        model = tf.keras.models.load_model(directory_model + model_path)
        input_size = (len(model.layers[0].output_shape[:]))
        print(f'Model path: {directory_model + model_path}, input size: {input_size}')

    return model, input_size


def get_img_array(img_path, size):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def load_preprocess_input(backbone_model):
    """

    :param backbone_model:
    :type backbone_model:
    :return:
    :rtype:
    """
    if backbone_model == 'vgg16':
        preprocessing_function = tf.keras.applications.vgg16.preprocess_input
        size = (224, 224)

    elif backbone_model == 'vgg19':
        preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        size = (224, 224)

    elif backbone_model == 'inception_v3':
        preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
        size = (299, 299)

    elif backbone_model == 'resnet50':
        preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        size = (224, 224)

    elif backbone_model == 'resnet101':
        preprocessing_function = tf.keras.applications.resnet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'mobilenet':
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'densenet121':
        preprocessing_function = tf.keras.applications.densenet.preprocess_input
        size = (224, 224)

    elif backbone_model == 'xception':
        preprocessing_function = tf.keras.applications.xception.preprocess_input
        size = (299, 299)
    else:
        raise ValueError(f' MODEL: {backbone_model} not found')

    return preprocessing_function, size


def create_auxiliar_networks(model, last_conv_layer_name, classifier_layer_names, cap_network=[]):
    """

    :param model:
    :type model:
    :param last_conv_layer_name:
    :type last_conv_layer_name:
    :param classifier_layer_names:
    :type classifier_layer_names:
    :param cap_network:
    :type cap_network:
    :return:
    :rtype:
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output, name='Features_last_layer_network')
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    #cap_network.summary()
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    if cap_network:
        # here is all the problem
        x = cap_network(x)

    classifier_model = tf.keras.Model(classifier_input, x, name='Classifier_Model')

    return last_conv_layer_model, classifier_model


def make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model):
    #last_conv_layer_model.summary()
    #classifier_model.summary()

    """
    :param img_array:
    :type img_array:
    :param last_conv_layer_model:
    :type last_conv_layer_model:
    :param classifier_model:
    :type classifier_model:
    :return:
    :rtype:
    """
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def generate_heat_map_and_mask(heatmap, img, img_size):
    """

    Parameters
    ----------
    heatmap : (array) heatmap generated with Gradcam
    img : (array) 3ch image
    img_size : (tuple) size of the input to the network

    Returns
    -------
    superimposed_img: (array)
    mask_heatmap: (array)
    binary_mask: (array)

    """
    mask_heatmap = cv2.resize(heatmap, img_size)

    if np.isnan(np.sum(np.unique(mask_heatmap))):
        mask_heatmap = np.zeros(img_size)
    else:
        limit = 0.7 * np.amax(mask_heatmap)
        mask_heatmap[mask_heatmap >= limit] = 1
        mask_heatmap[mask_heatmap < limit] = 0

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    w, d = np.shape(mask_heatmap)
    binary_mask = np.zeros((w, d, 3))
    binary_mask[:, :, 0] = mask_heatmap * 255
    binary_mask[:, :, 1] = mask_heatmap * 255
    binary_mask[:, :, 2] = mask_heatmap * 255

    return superimposed_img, mask_heatmap, binary_mask


def get_classifier_cap(model, name_backbone):
    save_layers = False
    layer_names = [layer.name for layer in model.layers]

    cap_model = keras.Sequential()
    for idx, l in enumerate(model.layers):
        if l.name == name_backbone:
            save_layers = True
        if save_layers is True and l.name != name_backbone:
            layer_idx = layer_names.index(l.name)
            #print(layer_idx, l.name)
            cap_model.add(model.layers[layer_idx])

    input_shape = (None, None, None, 2048)
    cap_model.build(input_shape)
    #cap_model.summary()
    return cap_model


def analyze_data_gradcam(directory_model, dataset_dir, dataset_to_analyze, output_dir='', plot=False,
                         save_results=True):
    """
    Given a classification network and a dataset to analyze, it returns the heat-map and the binary mask
    Parameters
    ----------
    name_model :
    dataset_dir (str): dataset_dir or video to analyze
    dataset_to_analyze :
    output_dir :
    plot (bool):
    save_results (bool):

    Returns
    -------

    """

    if os.path.isdir(dataset_to_analyze):
        if output_dir == '':
            gradcam_predictions_predictions_dir = ''.join([dataset_dir, '/gradcam_predictions/'])

            if not os.path.isdir(gradcam_predictions_predictions_dir):
                os.mkdir(gradcam_predictions_predictions_dir)

            output_dir = ''.join([dataset_dir,
                                  '/gradcam_predictions/'])

            heat_maps_dir = output_dir + 'heatmaps/'
            binary_masks_dir = output_dir + 'predicted_masks/'
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            if not os.path.isdir(heat_maps_dir):
                os.mkdir(heat_maps_dir)

            if not os.path.isdir(binary_masks_dir):
                os.mkdir(binary_masks_dir)

        model, _ = load_model(directory_model)
        model.summary()
        #backbone_model = model.get_layer(index=0)
        #classifier_cap = model.get_layer(index=-1)
        backbone_model = model.get_layer(index=4)# here you need to modify it to get only the first 8 layers
        classifier_cap = get_classifier_cap(model, 'resnet50')
        print(f'Backbone identified: {backbone_model.name}')
        classifier_layer_names = list()
        print(type(backbone_model))

        for l in reversed(model.layers):
            classifier_layer_names.append(l.name)

            if l.__class__.__name__ == 'Concatenate':
                last_conv_layer_name = l.name
                break

            if l.__class__.__name__ == 'Add':
                last_conv_layer_name = l.name
                break

            if l.__class__.__name__ == 'Conv2D':
                last_conv_layer_name = l.name
                break

        print(classifier_layer_names)
        #classifier_layer_names.remove(last_conv_layer_name)
        #classifier_layer_names.reverse()
        classifier_layer_names = ["avg_pool", "predictions"]
        last_conv_layer_name = 'conv5_block3_out'
        last_conv_layer_model, classifier_model = create_auxiliar_networks(model,
                                                            last_conv_layer_name,
                                                            classifier_layer_names,
                                                            cap_network=classifier_cap)

        #last_conv_layer = model.get_layer(last_conv_layer_name)
        # load the data
        test_dataset = os.listdir(dataset_to_analyze)
        test_dataset = [f for f in test_dataset if os.path.isdir(os.path.join(dataset_to_analyze, f))]
        list_imgs = os.listdir(dataset_to_analyze)
        list_imgs = [os.path.join(dataset_to_analyze, f) for f in list_imgs]
        #for folder in test_dataset:
        #    dir_folder = ''.join([dataset_to_analyze, folder, '/'])
        #    imgs_subdir = [dir_folder + f for f in os.listdir(dir_folder) if f.endswith('.png') or f.endswith('.jpg')]
        #    list_imgs = list_imgs + imgs_subdir

        for i, img_path in enumerate(tqdm.tqdm(list_imgs, desc=f'Making mask predictions, {len(list_imgs)} images')):
            # if you want to pick random paths uncomment bellow and comment the previous one to have a counter
            #img_path = random.choice(list_imgs)
            preprocess_input, img_size = load_preprocess_input('resnet50')
            img_array = preprocess_input(get_img_array(img_path, size=img_size))
            img_name = os.path.split(img_path)[-1]
            img = tf.keras.preprocessing.image.load_img(img_path)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.preprocessing.image.smart_resize(img, img_size, interpolation='bilinear')

            test_img = cv2.imread(img_path)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(test_img, img_size, interpolation=cv2.INTER_AREA)

            heatmap = make_gradcam_heatmap(img_array, last_conv_layer_model, classifier_model)
            superimposed_img, mask_heatmap, binary_mask = generate_heat_map_and_mask(heatmap, img, img_size)
            if save_results is True:
                cv2.imwrite(binary_masks_dir + img_name, binary_mask)
                superimposed_img.save(heat_maps_dir + img_name)

            if plot is True:
                plt.figure()
                plt.subplot(131)
                plt.imshow(img_resized)
                plt.subplot(132)
                plt.imshow(superimposed_img)
                plt.subplot(133)
                plt.imshow(mask_heatmap)
                plt.show()


def main(_argv):
    model_directory = FLAGS.model_directory
    dataset_dir = FLAGS.dataset_dir
    dataset_to_analyze = FLAGS.dataset_to_analyze
    analyze_data_gradcam(model_directory, dataset_dir, dataset_to_analyze, output_dir='', plot=False,
                         save_results=True)


if __name__ == '__main__':
    flags.DEFINE_string('model_directory', '', 'name of the model')
    flags.DEFINE_string('dataset_dir', '', 'name of the model')
    flags.DEFINE_string('dataset_to_analyze', '', 'name of the model')
    try:
        app.run(main)
    except SystemExit:
        pass
    pass