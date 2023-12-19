import time
import os
import load_data as dam
from models.classification_models import *
import datetime
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
import utils.data_analysis as daa
from absl import app, flags
from absl.flags import FLAGS


def eager_train(model, train_dataset, epochs, batch_size):
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                logits = model(x_batch_train, training=True)
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))


def custom_training(model_name, path_train_dataset, path_val_dataset, max_epochs, patience=15, batch_size=2,
                     learning_rate=0.0001, results_dir=os.path.join(os.getcwd(), 'results'), backbone_network='resnet101',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[],
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     path_test_data=''):

    train_dataset_dict = dam.load_dataset_from_directory(path_train_dataset)
    train_dataset = dam.make_tf_image_dataset(train_dataset_dict, training_mode=True, input_size=[224, 224],
                                          batch_size=batch_size)
    unique_classes = np.unique([train_dataset_dict[k]['class'] for k in train_dataset_dict.keys()])

    valid_dataset_dict = dam.load_dataset_from_directory(path_val_dataset)
    valid_dataset = dam.make_tf_image_dataset(valid_dataset_dict, training_mode=True, input_size=[224, 224],
                                              batch_size=batch_size)

    if model_name == 'simple_classifier':
        model = simple_classifier(len(unique_classes), backbone=backbone_network)
        model.summary()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    loss_fn = loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    backbone_model = backbone_network
    new_results_id = dam.generate_experiment_ID(name_model=model_name, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model)

    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'name model': 'semi_supervised_resnet101',
                              'backbone': backbone_model,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate)}

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exist create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    dam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            t_loss = loss_fn(y_true=labels, y_pred=predictions)
            print(t_loss)
        gradients = tape.gradient(t_loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(t_loss)
        train_accuracy_val = train_accuracy(labels, predictions)
        return train_loss_val, train_accuracy_val

    @tf.function
    def valid_step(images, labels):
        # pred_teacher = teacher_model(images, training=False)
        # labels = tf.argmax(pred_teacher, axis=1)
        predictions = model(images, training=False)
        v_loss = loss_fn(labels, predictions)
        val_loss = valid_loss(v_loss)
        val_acc = valid_accuracy(labels, predictions)

        return val_loss, val_acc

    @tf.function
    def prediction_step(images):
        predictions = model(images, training=False)
        return predictions

    patience = patience
    wait = 0
    # start training
    best_loss = 999
    start_time = datetime.datetime.now()
    epoch_counter = list()
    train_loss_list = list()
    train_accuracy_list = list()

    model_dir = os.path.join(results_directory, 'model_weights')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])

    for epoch in range(max_epochs):
        epoch_counter.append(epoch)
        train_loss_list.append(train_loss.result().numpy())
        train_accuracy_list.append(train_accuracy.result().numpy())

        t = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0

        template = 'ETA: {} - epoch: {} loss: {:.5f}  acc: {:.5f}'
        for x, train_labels in train_dataset:
            step += 1
            images = x
            train_loss_value, t_acc = train_step(images, train_labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            print(template.format(round((time.time() - t) / 60, 2), epoch + 1, train_loss_value,
                                  float(train_accuracy.result())))

        for x, valid_labels in valid_dataset:
            valid_images = x
            valid_step(valid_images, valid_labels)
            with val_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  max_epochs,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

        # checkpoint.save(epoch)
        # writer.flush()

        wait += 1
        if epoch == 0:
            best_loss = valid_loss.result()
        if valid_loss.result() < best_loss:
            best_loss = valid_loss.result()
            # model.save_weights('model.h5')
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break

    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')
    print('Total Training TIME:', (datetime.datetime.now() - start_time))

    if path_test_data:

        print(f'Making predictions on test dataset: {path_test_data}')
        # 2Do load saved model

        test_dataset_dict = dam.load_dataset_from_directory(path_test_data)
        test_dataset = dam.make_tf_image_dataset(test_dataset_dict, training_mode=False, input_size=[224, 224], batch_size=1)

        list_images = [test_dataset_dict[x]['Frame_id'] for x in test_dataset_dict.keys()]
        list_labels = [test_dataset_dict[x]['class'] for x in test_dataset_dict.keys()]
        unique_class = list(np.unique([train_dataset_dict[k]['class'] for k in train_dataset_dict.keys()]))
        test_labels = [unique_class.index(x) for x in list_labels]

        list_predictions = list()
        i = 0
        for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions on test dataset')):
            x = data_batch[0]
            img = x[0]
            image = tf.expand_dims(img, axis=0)
            pred = prediction_step(image)
            prediction = list(pred.numpy()[0])
            prediction_label = unique_class[prediction.index(np.max(prediction))]
            list_predictions.append(prediction_label)
            i += 1

        header_column = list()
        header_column.insert(0, 'img name')
        header_column.append('real label')
        header_column.append('predicted label')

        df = pd.DataFrame(list(zip(list_images, list_labels, list_predictions)), columns=header_column)

        path_results_csv_file = os.path.join(results_directory, 'predictions.csv')
        df.to_csv(path_results_csv_file, index=False)

        print(f'csv file with results saved: {path_results_csv_file}')

        dir_conf_matrix = os.path.join(results_directory, 'confusion_matrix.png')
        daa.compute_confusion_matrix(list_labels, list_predictions, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix)


def main(_argv):

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    path_dataset = FLAGS.path_train_dataset
    path_val_dataset = FLAGS.path_val_dataset
    type_training = FLAGS.type_training
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    results_dir = FLAGS.results_dir
    learning_rate = FLAGS.learning_rate
    backbone = FLAGS.backbone
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = ["accuracy", tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]

    if type_training == 'eager_training':
        dataset_dict = dam.load_dataset_from_directory(path_dataset)
        unique_classes = np.unique([dataset_dict[k]['class'] for k in dataset_dict.keys()])
        tf_ds = dam.make_tf_image_dataset(dataset_dict, training_mode=True, input_size=[224, 224], batch_size=batch_size)
        model = simple_classifier(len(unique_classes))
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        eager_train(model, tf_ds, epochs, batch_size)

    elif type_training == 'custom_training':

        list_folders = [f for f in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, f))]
        if path_val_dataset:
            path_validation_dataset = path_val_dataset
            path_train_dataset = path_dataset
        elif 'val' in list_folders:
            path_validation_dataset = os.path.join(path_dataset, 'val')
            path_train_dataset = os.path.join(path_dataset, 'train')
        else:
            path_train_dataset = path_dataset
            path_validation_dataset = path_dataset

        custom_training(name_model, path_train_dataset, path_validation_dataset, epochs, patience=15, batch_size=batch_size, backbone_network=backbone,
                         loss=loss, metrics=metrics, optimizer=optimizer, path_test_data=path_dataset)
    else:
        print(f'{type_training} not in options!')


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('path_train_dataset', '', 'directory training dataset')
    flags.DEFINE_string('path_val_dataset', '', 'directory validation dataset')
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