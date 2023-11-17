from absl import app, flags
from absl.flags import FLAGS
import os
import tensorflow as tf
import datetime
from models.model_utils import build_rcnn
import load_data as ld
import numpy as np


def main(_argv):

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    path_dataset = FLAGS.path_data
    name_model = FLAGS.model
    path_train_annotations = FLAGS.path_train_dataset
    path_val_annotations = FLAGS.path_val_dataset
    epochs = FLAGS.epochs
    num_frames_per_clip = FLAGS.num_frames_per_clip
    batch_size = FLAGS.batch_size

    dicta_train = ld.load_dataset_from_directory(path_dataset, path_train_annotations)
    dicta_val = ld.load_dataset_from_directory(path_dataset, path_val_annotations)
    list_files = list(dicta_train.keys())
    unique_classes = list(np.unique([dicta_train[img_id]['Bleeding'] for img_id in list_files]))

    if name_model == 'rcnn':

        output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.int16))

        train_ds = tf.data.Dataset.from_generator(
            ld.FrameGenerator(dicta_train, num_frames_per_clip, training=True, num_of_no_events=1.0),
            output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(
            ld.FrameGenerator(dicta_val, num_frames_per_clip, training=True, num_of_no_events=1.0),
            output_signature=output_signature)

        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

        train_frames, train_labels = next(iter(train_ds))
        print(f'Shape of training set of frames: {train_frames.shape}')
        print(f'Shape of training labels: {train_labels.shape}')

        model = build_rcnn(len(unique_classes))

        start_time = datetime.datetime.now()
        model.fit(train_ds,
                  epochs=epochs,
                  validation_data=val_ds,
                  callbacks=tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'))
        print('Total Training TIME:', (datetime.datetime.now() - start_time))


if __name__ == '__main__':

    flags.DEFINE_string('model', '', 'name of the model')
    flags.DEFINE_string('path_data', '', 'directory training dataset')
    flags.DEFINE_string('path_train_dataset', '', 'directory training dataset')
    flags.DEFINE_string('path_val_dataset', '', 'directory validation dataset')
    flags.DEFINE_integer('num_frames_per_clip', 20, 'number of frames per each video clip')
    flags.DEFINE_string('type_training', '', 'eager_train or custom_training')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epochs', 1, 'epochs')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('analyze_data', True, 'analyze the data after the experiment')
    flags.DEFINE_string('backbone', 'resnet101', 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')
    try:
        app.run(main)
    except SystemExit:
        pass