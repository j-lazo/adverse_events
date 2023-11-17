import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import datetime
import pickle
import time
import numpy as np
import pandas as pd
import tqdm
import shutil
import random
import copy
import tensorflow_addons as tfa
from collections import ChainMap
import cv2
from matplotlib import pyplot as plt


def load_dataset_from_directory(path_frames, path_annotations, items_to_use=['Overall', 'Bleeding'],
                                class_condition=''):
    """
        Give a path, creates two lists with the
        Parameters
        ----------
        path_dataset (Str): the absolute path to the dataset. The directory should be built as:
            |
            |-- directory
                | -- frames
                    | -- case 1
                        | -- frame1.jpg
                        | -- frame2.jpg
                        |...
                    |-- case 2
                    ...
                    |-- case n

                | -- labels
                    | -- train
                        | -- annotations_train.pickle
                    | -- val
                        | -- annotations_val.pickle
                    | -- test
                        | -- annotations_test.pickle

        if the labels are divided by folds, they should have the same name or some ID inside the name

        Returns (dict):  single dictionary of dictionaries with all the entries of frames per case
        -------

        """
    output_dict = {}
    pickleFile = open(path_annotations, "rb")
    pickleInfo = pickle.load(open(path_annotations, "rb"))
    # read dictionary of list (cases) each list has a dictionary of frames
    list_cases = pickleInfo.keys()
    for case in tqdm.tqdm(list_cases, desc=f"Loading data"):
        annotated_frames = pickleInfo[case]
        frames_IDs = [d['Frame_id'] for d in annotated_frames]
        path_case = os.path.join(path_frames, case)
        list_imgs = os.listdir(path_case)
        # double-check the list since not all the frames have annotations
        list_imgs = [f.split('.')[0] for f in list_imgs if f.split('.')[0] in frames_IDs]
        list_real_imgs = [f for f in annotated_frames if
                          os.path.isfile(os.path.join(path_frames, case, f['Frame_id'] + '.jpg'))]
        dict_frames = {case + d['Frame_id']: {'case_id': case,
                                              'Frame_id': d['Frame_id'],
                                              'Path_img': os.path.join(path_frames, case, d['Frame_id'] + '.jpg'),
                                              'Overall': d['Overall'],
                                              'Bleeding': d['Bleeding'],
                                              'Event_ID': d['Event_ID']
                                              } for d in annotated_frames if d['Frame_id'] in list_imgs}
        if class_condition:
            dict_frames = {case + d['Frame_id']: {'case_id': case,
                                                  'Frame_id': d['Frame_id'],
                                                  'Path_img': os.path.join(path_frames, case, d['Frame_id'] + '.jpg'),
                                                  'Overall': d['Overall'],
                                                  'Bleeding': d['Bleeding'],
                                                  'Event_ID': d['Event_ID']
                                                  } for d in annotated_frames if d['Frame_id'] in list_imgs and
                           d[class_condition] == 1 and os.path.isfile(
                os.path.join(path_frames, case, d['Frame_id'] + '.jpg'))}

        output_dict = {**output_dict, **dict_frames}
        # option b with ChainMap, check which one is more efficient
        # output_dict = dict(ChainMap({}, output_dict, dict_frames))

    print(f'Dataset with {len(output_dict)} elements')
    return output_dict


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

#def make_tf_clips_dataset(dictionary_labels, batch_size=2, training_mode=False,
#                    num_repeat=None, custom_training=False, ignore_labels=False, specific_iaes=[], n=40):
#    return 0


def add_reversed_elements(input_list, n):
    reversed_list = input_list[::-1]
    while len(input_list) < n:
        input_list.extend(reversed_list)
        reversed_list = input_list[::-1]

    return input_list[:n]


def split_list(list_images, n_lim, min_left=15, list_pad=None):
    chunks = [list_images[x:x + n_lim] for x in range(0, len(list_images), n_lim)]
    for chunk in chunks:
        if len(chunk) < min_left:
            chunks.remove(chunk)
        elif len(chunk) >= min_left and len(chunk) < n_lim:
            if list_pad == 'zeros':
                extra = np.zeros([n_lim - len(chunk)])
                chunk.extend(extra)
            elif list_pad == None:
                extra = [None] * (n_lim - len(chunk))
                chunk.extend(extra)
            elif list_pad == 'mirror':
                idx = chunks.index(chunk)
                chunks[idx] = add_reversed_elements(chunk, n_lim)

    return chunks


def decode_image(file_name):
    image = tf.io.read_file(file_name)
    if tf.io.is_jpeg(image):
        image = tf.io.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_png(image, channels=3)

    return image


def format_frames(frame, output_size):
    """
      Pad and resize an image from a video.

      Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

      Return:
        Formatted frame with padding of specified output size.
    """
    frame = decode_image(frame)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def load_image_batch(list_images, output_size=(224, 224)):
    output_tensor = tf.zeros([len(list_images), *output_size, 3], dtype=tf.float32)

    for j, img_path in enumerate(list_images):
        img = format_frames(img_path, output_size)
        index = tf.constant([[j]])
        output_tensor = tf.tensor_scatter_nd_update(output_tensor, index, tf.expand_dims(img, axis=0))

    return output_tensor


class FrameGenerator:
    def __init__(self, dictionary_labels, n_frames, training=False, output_size=(224, 224),
                 specific_events=['Bleeding'],
                 num_of_no_events=0):
        """ Returns a set of frames with their associated label.

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """

        self.dictionary_labels = dictionary_labels
        self.n_frames = n_frames
        self.training = training
        self.list_files = list(dictionary_labels.keys())
        self.output_size = output_size
        self.specific_events = specific_events
        self.num_of_no_events = num_of_no_events

        # get all the unique ID's of all the events (sometimes there is more than one event per image)
        events_ids = [self.dictionary_labels[img_id]['Event_ID'] for img_id in self.list_files]
        # to unpack all the list of list, into a single list
        all_events = [e for list_events in events_ids for e in list_events]
        # to get the unique IDs
        unique_events = np.unique(all_events)
        self.dict_events_frames = {e: list() for e in unique_events}
        self.dict_events_labels = {e: list() for e in unique_events}

        for event in unique_events:
            for img_id in self.list_files:
                if event in self.dictionary_labels[img_id]['Event_ID']:
                    for event_type in specific_events:
                        if self.dictionary_labels[img_id][event_type] == 1:
                            self.dict_events_frames[event].append(self.dictionary_labels[img_id]['Path_img'])
                            self.dict_events_labels[event].append(self.dictionary_labels[img_id][event_type])

        list_all_events = list()
        list_all_labels = list()

        for event in self.dict_events_frames.keys():
            list_events = split_list(self.dict_events_frames[event], n_frames, list_pad='mirror')
            list_labels = split_list(self.dict_events_labels[event], n_frames, list_pad='mirror')
            for j, clip in enumerate(list_events):
                list_all_events.append(clip)
                list_all_labels.append(list_labels[j][0])

        for j, label in enumerate(list_all_labels):
            if label is None:
                label[j] = 0

        list_of_no_events = list()
        list_of_no_labels = list()
        temp_list = list()

        if isinstance(self.num_of_no_events, int):
            num_no_events = self.num_of_no_events
        else:
            num_no_events = int(self.num_of_no_events * len(list_all_events))

        while len(list_of_no_events) < num_no_events:
            selected = random.choice(self.list_files)
            if self.dictionary_labels[selected]['Overall'] == 0:
                idx = self.list_files.index(selected)
                temp_list = list(
                    x for x in self.list_files[idx:idx + n_frames] if self.dictionary_labels[x]['Overall'] == 0)
                if len(temp_list) < n_frames:
                    temp_list = add_reversed_elements(temp_list, n_frames)
            list_of_no_events.append([self.dictionary_labels[x]['Path_img'] for x in temp_list])
            list_of_no_labels.append([self.dictionary_labels[x]['Overall'] for x in temp_list])

        list_of_no_labels = [x[0] for x in list_of_no_labels]

        # list_path_of_no_events = [self.dictionary_labels[x]['Path_img'] for x in temp_list]
        # list_of_no_labels = [self.dictionary_labels[x]['Overall'] for x in temp_list]

        list_all_events += list_of_no_events
        list_all_labels += list_of_no_labels
        #list_all_labels = tf.data.Dataset.from_tensor_slices(list_all_labels)
        self.pairs = list(zip(list_all_events, list_all_labels))

    def __call__(self):
        if self.training:
            random.shuffle(self.pairs)

        for paths, labels in self.pairs:
            image_batch = load_image_batch(paths, self.output_size)
            yield image_batch, labels

def make_tf_image_dataset(dictionary_labels, labels_list, batch_size=2, training_mode=False,
                    num_repeat=None, custom_training=False, ignore_labels=False):

    list_files = list(dictionary_labels.keys())

    def decode_image(file_name):
        image = tf.io.read_file(file_name)
        if tf.io.is_jpeg(image):
            image = tf.io.decode_jpeg(image, channels=3)
        else:
            image = tf.image.decode_png(image, channels=3)

        return image

    def rand_degree(lower, upper):
        return random.uniform(lower, upper)

    def rotate_img(img, lower=0, upper=180):

        upper = upper * (np.pi / 180.0)  # degrees -> radian
        lower = lower * (np.pi / 180.0)
        img = tf.keras.layers.RandomRotation(rand_degree(lower, upper), fill_mode='nearest')(img)
        return img

    def parse_image(filename):

        image = decode_image(filename)
        image = tf.image.resize(image, [256, 256])

        if training_mode:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            #image = rotate_img(image)

        return image

    def configure_for_performance(dataset):
      dataset = dataset.shuffle(buffer_size=1000)
      dataset = dataset.batch(batch_size)
      if num_repeat:
          dataset = dataset.repeat()
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset

    path_imgs = list()
    images_class = list()
    img_paths = list()

    if training_mode:
        random.shuffle(list_files)

    for img_id in list_files:
        path_imgs.append(dictionary_labels[img_id]['Path_img'])
        images_class.append(dictionary_labels[img_id]['Bleeding'])

    filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    unique_classes = list(np.unique(images_class))
    num_classes = len(unique_classes)
    labels = [unique_classes.index(v) for v in images_class]
    network_labels = list()

    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    #path_files = tf.data.Dataset.from_tensor_slices()

    ds = tf.data.Dataset.zip((images_ds, labels_ds), filenames_ds)

    if training_mode:
        ds = configure_for_performance(ds)
    else:
        ds = ds.batch(batch_size)

    print(f'TF dataset with {len(path_imgs)} elements')
    return ds, num_classes


#def analyze_video_dataset(dictionary_labels):
#    labels = ['normal', 'bleeding']
#    tf_dataset = make_tf_clips_dataset(dictionary_labels)


def analyze_dataset(tf_dataset):
    labels = ['normal', 'bleeding']
    for pack, z in tf_dataset:

        print(z)
        x = pack[0]
        y = pack[1]

        img1 = x[0].numpy()
        img1 *= (255.0/img1.max())
        img2 = x[1].numpy()
        img2 *= (255.0 / img2.max())
        label1 = y[0].numpy()
        label2 = y[1].numpy()

        fig = plt.figure()

        # Needed to add spacing between 1st and 2nd row
        # Add a margin between the main title and sub-plots
        fig.subplots_adjust(hspace=0.4, top=0.85)

        # Add the main title
        fig.suptitle("Sample Batch", fontsize=15)

        # Add the subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.imshow(img1.astype('int'))
        ax2.imshow(img2.astype('int'))

        # Add the text for each subplot
        ax1.title.set_text(labels[label1] + ' ' + str(y[0].numpy()))
        ax2.title.set_text(labels[label2] + ' ' + str(y[1].numpy()))
        plt.show()
