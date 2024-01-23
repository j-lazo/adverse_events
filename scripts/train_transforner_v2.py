import time
import os
import utils.data_management as dam
from models.classification_models import *
from models.transformer import *
import datetime
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
import utils.data_analysis as daa
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import Progbar

def correct_labels(list_labels):
    out_list = [s.replace('-', ' ') for s in list_labels]
    return out_list


def model_fit(model_name, train_dataset, valid_dataset, max_epochs, fold, input_shape, image_size,
              patch_size, num_patches, projection_dim, transformer_layers, num_heads, transformer_units, mlp_head_units,
              num_classes, patience=15, batch_size=2, learning_rate=0.0001, results_dir=os.path.join(os.getcwd(), 'results'),
              loss={'classi': tf.keras.losses.SparseCategoricalCrossentropy(),
                    'grading': tf.keras.losses.SparseCategoricalCrossentropy()}, metrics=[],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              test_dataset=None, output_type=''):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print('model name:', model_name)
    if model_name == 'multi_output_transformer':

        model = create_multi_output_vit_classifier(input_shape, image_size, patch_size, num_patches, projection_dim,
                          transformer_layers, num_heads, transformer_units, mlp_head_units,
                          num_classes)
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # ID name for the folder and results
    new_results_id = dam.generate_experiment_ID(name_model=model_name, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model='_')

    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'training type': 'fit model',
                              'name model': model_name,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'output type': output_type,
                              'fold': fold,
                              }

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exist create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    else:
        count = 0
        while os.path.isdir(results_directory):
            results_directory = ''.join([results_dir, '/', new_results_id, '-', str(count), '/'])
            count += 1
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    dam.save_yaml(path_experiment_information, information_experiment)

    model_dir = os.path.join(results_directory, 'saved_model.h5')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])

    temp_name_model = os.path.join(model_dir, 'best_model_.h5')
    history_name = ''.join([results_directory, 'train_history_', new_results_id, "_.csv"])
    callbacks = [ModelCheckpoint(temp_name_model, monitor="val_loss", save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', patience=15),
                CSVLogger(history_name),
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    start_time = datetime.datetime.now()
    trained_mode = model.fit(x=train_dataset, validation_data=valid_dataset,
                             epochs=max_epochs, verbose=1, callbacks=callbacks)

    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')
    print('Total Training TIME:', (datetime.datetime.now() - start_time))

    # Now test in the test dataset
    list_images = list()
    list_predictions = list()
    list_labels = list()
    for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions on test dataset')):
        image_labels = data_batch[0]
        path_img = data_batch[1]
        image, labels = image_labels
        pred = model.predict(image)
        prediction = list(pred[0]).index(max(list(pred[0])))
        list_images.append(path_img.numpy()[0].decode("utf-8"))
        list_predictions.append(prediction)
        gt_label = list(labels.numpy()[0]).index(max(list(labels.numpy()[0])))
        list_labels.append(gt_label)

    header_column = list()
    header_column.insert(0, 'img name')
    header_column.append('label')
    header_column.append('predicted')

    df = pd.DataFrame(list(zip(list_images, list_labels, list_predictions)), columns=header_column)

    path_results_csv_file = os.path.join(results_directory, 'predictions.csv')
    df.to_csv(path_results_csv_file, index=False)

    print(f'csv file with results saved: {path_results_csv_file}')

    dir_conf_matrix_phase = os.path.join(results_directory, 'confusion_matrix.png')
    daa.compute_confusion_matrix(list_labels, list_predictions, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_phase)


def custom_training(model_name, train_dataset, valid_dataset, max_epochs, fold, input_shape, image_size,
              patch_size, num_patches, projection_dim, transformer_layers, num_heads, transformer_units, mlp_head_units,
              num_classes, patience=15, batch_size=2, learning_rate=0.0001, results_dir=os.path.join(os.getcwd(), 'results'),
              loss={'classi': tf.keras.losses.SparseCategoricalCrossentropy(),
                    'grading': tf.keras.losses.SparseCategoricalCrossentropy()}, metrics=[],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              test_dataset=None, output_type='', verbose=False, gpus_available=None):

    loss_fn_1 = loss['classi']
    loss_fn_2 = loss['grading']

    #@tf.function
    def train_step(images, labels, ratio_grade_loss=1.0):
        with tf.GradientTape() as tape:
            labels_1, labels_2 = labels
            predictions_1, predictions_2 = model(images, training=True)
            #print(np.shape(predictions_1.numpy()), 'predictions 1')
            #print(np.shape(predictions_2.numpy()), 'predictions 2')
            #print(np.shape(labels_1), 'label 1')
            #print(np.shape(labels_2), 'label 2')
            #print('predictions 1:', predictions_1.numpy())
            #print('labels 1:', labels_1)
            t_loss_1 = loss_fn_1(y_true=labels_1, y_pred=predictions_1)
            #print('loss 1 ok')
            t_loss_2 = loss_fn_2(y_true=labels_2, y_pred=predictions_2)
            #print('predictions 2:', predictions_2.numpy())
            #print('labels 2:', labels_2)
            #print('loss 2 ok')
            t_loss = t_loss_1 + ratio_grade_loss * t_loss_2
            #print('loss 1', t_loss_1)
            #print('loss 2', t_loss_2)
            #print('loss t', t_loss)
        gradients = tape.gradient(t_loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss_val = train_loss(t_loss)

        #predictions_acc_1 = tf.argmax(predictions_1, axis=1)
        #predictions_acc_2 = tf.argmax(predictions_2, axis=1)
        #print('Accuracies')
        #print('labels 1', labels_1)
        #print('predictions', predictions_1)
        train_accuracy_val_1 = train_accuracy_1(labels_1, predictions_1)
        train_accuracy_val_2 = train_accuracy_2(labels_2, predictions_2)
        class_f1 = [f1_score(labels_1[:, i], np.argmax(predictions_1, axis=-1)[:, i], zero_division=0.0) for i in range(4)]
        #grade_f1 = [f1_score(labels_2[:, i], np.argmax(predictions_2, axis=-1)[:, i], average='macro') for i in range(4)]
        grade_mse = [mean_squared_error(labels_2[:, i], np.argmax(predictions_2, axis=-1)[:, i]) for i in range(4)]
        value_train_loss_class = train_loss_classi(t_loss_1)
        value_train_loss_grading = train_loss_grading(t_loss_2)
        losses = [train_loss_val, value_train_loss_class, t_loss_2]
        metrics = [train_accuracy_val_1, train_accuracy_val_2, class_f1, grade_mse]

        #print('f-1 class', class_f1)
        #avg_class_f1 = np.mean(class_f1)
        #print('f-1 average class', avg_class_f1)
        #print('MSE grade', grade_mse)
        #avg_grade_mse = np.mean(grade_mse)
        #print('f-1 average grade:', avg_grade_mse)

        return losses, metrics


    #@tf.function
    def make_f1_score_array(labels, predictions):
        f1_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in range(4):
            f1_scores.write(f1_score(labels[:, i], np.argmax(predictions, axis=-1)[:, i], zero_division=0.0))
        return f1_scores

    #@tf.function
    def valid_step(images, labels):
        labels_1, labels_2 = labels
        predictions_1, predictions_2 = model(images, training=True)
        v_loss_1 = loss_fn_1(y_true=labels_1, y_pred=predictions_1)
        v_loss_2 = loss_fn_2(y_true=labels_2, y_pred=predictions_2)
        v_loss = tf.reduce_mean(v_loss_1) + tf.reduce_mean(v_loss_2)
        val_loss = valid_loss(v_loss)
        #predictions_acc_1 = tf.argmax(predictions_1, axis=1)
        #predictions_acc_2 = tf.argmax(predictions_2, axis=1)
        val_accuracy_val_1 = valid_accuracy_1(labels_1, predictions_1)
        val_accuracy_val_2 = valid_accuracy_2(labels_2, predictions_2)
        val_class_f1 = [f1_score(labels_1[:, i], np.argmax(predictions_1, axis=-1)[:, i], zero_division=0.0) for i in range(4)]
        #val_class_f1 = make_f1_score_array(labels_1, predictions_1)
        val_grade_mse = [mean_squared_error(labels_2[:, i], np.argmax(predictions_2, axis=-1)[:, i]) for i in range(4)]
        #losses = tf.stack([val_loss, v_loss_1, v_loss_2])
        #metrics = tf.stack([val_accuracy_val_1, val_accuracy_val_2, val_class_f1, val_grade_mse])
        value_val_loss_classi = val_loss_classi(v_loss_1)
        value_val_liss_grading = val_loss_grading(v_loss_2)
        losses = [val_loss, value_val_loss_classi, value_val_liss_grading]
        metrics = [val_accuracy_val_1, val_accuracy_val_2, val_class_f1, val_grade_mse]

        return losses, metrics

    @tf.function
    def prediction_step(images):
        predictions = model(images, training=False)
        return predictions

    if model_name == 'multi_output_transformer':

        model = create_multi_output_vit_classifier(input_shape, image_size, patch_size, num_patches, projection_dim,
                          transformer_layers, num_heads, transformer_units, mlp_head_units,
                          num_classes)
        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    elif model_name == 'two_outputs_classifier':
        pass

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_classi = tf.keras.metrics.Mean(name='train_loss_classi')
    train_loss_grading = tf.keras.metrics.Mean(name='train_loss_grading')
    train_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_1')
    train_accuracy_2 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_2')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    val_loss_classi = tf.keras.metrics.Mean(name='train_loss_classi')
    val_loss_grading = tf.keras.metrics.Mean(name='train_loss_grading')
    valid_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy_1')
    valid_accuracy_2 = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy_2')

    f1_score_metric_train = tf.keras.metrics.Mean(name='train_f1')
    mse_score_metric_train = tf.keras.metrics.Mean(name='train_mse')

    f1_score_metric_val = tf.keras.metrics.Mean(name='val_f1')
    mse_score_metric_val = tf.keras.metrics.Mean(name='val_mse')

    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # ID name for the folder and results
    new_results_id = dam.generate_experiment_ID(name_model=model_name, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model='_')

    # the information needed for the yaml
    training_date_time = datetime.datetime.now()
    information_experiment = {'experiment folder': new_results_id,
                              'date': training_date_time.strftime("%d-%m-%Y %H:%M"),
                              'training type': 'custom training',
                              'name model': model_name,
                              'batch size': int(batch_size),
                              'learning rate': float(learning_rate),
                              'output type': output_type,
                              'fold': fold,
                              }

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exist create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    else:
        count = 0
        while os.path.isdir(results_directory):
            results_directory = ''.join([results_dir, '/', new_results_id, '-', str(count), '/'])
            count += 1
        os.mkdir(results_directory)

    path_experiment_information = os.path.join(results_directory, 'experiment_information.yaml')
    dam.save_yaml(path_experiment_information, information_experiment)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(results_directory, 'summaries', 'val'))

    patience = patience
    wait = 0
    # start training
    best_loss = 999
    start_time = datetime.datetime.now()
    epoch_counter = list()
    train_loss_list = list()
    train_classi_loss_list = list()
    train_grading_loss_list = list()
    val_loss_list = list()
    val_classi_loss_list = list()
    val_grading_loss_list = list()
    train_accuracy_list_1 = list()
    train_accuracy_list_2 = list()
    val_accuracy_list_1 = list()
    val_accuracy_list_2 = list()

    f1_train_classi_list = list()
    f1_train_avg_classi_list = list()
    mse_train_grading_list = list()
    mse_train_avg_grading_list = list()

    f1_val_classi_list = list()
    f1_val_avg_classi_list = list()
    mse_val_grading_list = list()
    mse_val_avg_grading_list = list()

    model_dir = os.path.join(results_directory, 'model_weights')
    os.mkdir(model_dir)
    model_dir = ''.join([model_dir, '/saved_weights'])

    # headers pd Dataframe
    header_column = list()
    header_column.insert(0, 'epoch')
    header_column.append('total train loss')
    header_column.append('class train loss')
    header_column.append('grade train loss')
    header_column.append('total val loss')
    header_column.append('class val loss')
    header_column.append('grade val loss')
    header_column.append('train acc classification')
    header_column.append('train acc grading')
    header_column.append('val acc class')
    header_column.append('val acc grading')
    header_column.append('train f-1 class')
    header_column.append('train MSE grading')
    header_column.append('val f-1 class')
    header_column.append('val MSE grading')

    #template = ('ETA: {} - epoch: {} loss: {:.5f}  acc classification: {:.5f}, acc grading: {:.5f}, '
    #            'F-1 classification: {}, F-1 avg: {:.5f}, MSE grading: {}, MSE avg: {.5.f}')

    for epoch in range(max_epochs):
    #    pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)
    #    for j in range(num_training_samples // batch_size):
    #        time.sleep(0.3)
    #        values = [('acc', np.random.random(1)), ('pr', np.random.random(1))]
    #        pb_i.add(batch_size, values=values)

        epoch_counter.append(epoch)
        train_loss_list.append(train_loss.result().numpy())
        train_classi_loss_list.append(train_loss_classi.result().numpy())
        train_grading_loss_list.append(train_loss_grading.result().numpy())
        train_accuracy_list_1.append(train_accuracy_1.result().numpy())
        train_accuracy_list_2.append(train_accuracy_2.result().numpy())

        f1_train_classi_list.append(f1_score_metric_train.result().numpy())
        mse_train_grading_list.append(mse_score_metric_train.result().numpy())
        f1_val_classi_list.append(f1_score_metric_val.result().numpy())
        mse_val_grading_list.append(mse_score_metric_val.result().numpy())

        val_loss_list.append(valid_loss.result().numpy())
        val_classi_loss_list.append(val_loss_classi.result().numpy())
        val_grading_loss_list.append(val_loss_grading.result().numpy())
        val_accuracy_list_1.append(valid_accuracy_1.result().numpy())
        val_accuracy_list_2.append(valid_accuracy_2.result().numpy())

        t = time.time()
        train_loss.reset_states()
        train_loss_classi.reset_states()
        train_loss_grading.reset_states()
        train_accuracy_1.reset_states()
        train_accuracy_2.reset_states()

        valid_loss.reset_states()
        valid_accuracy_1.reset_states()
        valid_accuracy_2.reset_states()
        val_loss_classi.reset_states()
        val_loss_grading.reset_states()

        f1_score_metric_train.reset_states()
        f1_score_metric_val.reset_states()
        mse_score_metric_train.reset_states()
        mse_score_metric_val.reset_states()

        step = 0
        metrics_names_train = ['train epoch', 'total loss', 'loss classification', 'loss grading', 'acc classification',
                         'acc grading', 'F-1 avg', 'MSE avg']

        for i, (x, train_labels) in enumerate(train_dataset):
            progbar_train = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=metrics_names_train)
            step += 1
            images = x
            train_loses, train_metrics = train_step(images, train_labels)
            values = [('train epoch', int(epoch + 1)), ('total loss', train_loss.result()),
                      ('loss classification', train_loss_classi.result()), ('loss grading', train_loss_grading.result()),
                      ('acc classification', float(train_accuracy_1.result())), ('acc grading', float(train_accuracy_1.result())),
                      ('F-1 avg', np.mean(train_metrics[2])), ('MSE avg', np.mean(train_metrics[3]))]
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('loss classification', train_loss_classi.result(), step=epoch)
                tf.summary.scalar('loss grading', train_loss_grading.result(), step=epoch)
                tf.summary.scalar('accuracy classification', train_accuracy_1.result(), step=epoch)
                tf.summary.scalar('accuracy grading', train_accuracy_2.result(), step=epoch)
            progbar_train.update(i, values=values)
            if verbose:
                #print(template.format(round((time.time() - t) / 60, 2), epoch + 1, train_loss.result(),
                #                      float(train_accuracy_1.result()), float(train_accuracy_1.result()),
                #                      train_metrics[2], np.mean(train_metrics[2]), train_metrics[3],
                #                      np.mean(train_metrics[3])))
                print(f'ETA: {round((time.time() - t) / 60, 2)} - epoch: {epoch + 1} loss: {train_loss.result():.5f},'
                      f'loss classification training: {train_loss_classi.result():.5f}, grading loss training: {train_loss_grading.result():.5f}'
                      f'acc classification: {float(train_accuracy_1.result()):.5f}, '
                      f'acc grading: {float(train_accuracy_1.result()):.5f}, '
                                          f'F-1 classification: {train_metrics[2]}, '
                      f'F-1 avg: {np.mean(train_metrics[2]):.5f}, MSE grading: {train_metrics[3]}, '
                      f'MSE avg: {np.mean(train_metrics[3]):.5f}')

        print(f"Epoch {epoch + 1}/{max_epochs + 1}. Total train loss {train_loss.result():.5f}, "
              f"loss classification training: {train_loss_classi.result():.5f}, grading loss training: {train_loss_grading.result():.5f},"
              f"acc classification: {float(train_accuracy_1.result()):.5f}, acc grading: {float(train_accuracy_1.result()):.5f}, "
              f"f'F-1 classification: {train_metrics[2]}, F-1 avg: {np.mean(train_metrics[2]):.5f}, MSE grading: {train_metrics[3]}, "
              f"MSE avg: {np.mean(train_metrics[3]):.5f}")

        metrics_names_val = ['val epoch', 'total loss', 'loss classification', 'loss grading', 'acc classification',
                               'acc grading', 'F-1 avg', 'MSE avg']

        for j, (x, valid_labels) in enumerate(valid_dataset):
            valid_images = x
            val_losses, val_metrics = valid_step(valid_images, valid_labels)
            progbar_val = tf.keras.utils.Progbar(len(valid_dataset), stateful_metrics=metrics_names_val)

            values = [('val epoch', int(epoch + 1)), ('total loss', valid_loss.result()),
                      ('loss classification', val_loss_classi.result()),
                      ('loss grading', val_loss_grading.result()),
                      ('acc classification', float(valid_accuracy_1.result())),
                      ('acc grading', float(valid_accuracy_2.result())),
                      ('F-1 avg', np.mean(val_metrics[2])), ('MSE avg', np.mean(val_metrics[3]))]

            progbar_val.add(j, values=values)
            with val_summary_writer.as_default():
                tf.summary.scalar('val loss', valid_loss.result(), step=epoch)
                tf.summary.scalar('val loss grading', val_loss_grading.result(), step=epoch)
                tf.summary.scalar('val loss classification', val_loss_classi.result(), step=epoch)
                tf.summary.scalar('val accuracy classification', valid_accuracy_1.result(), step=epoch)
                tf.summary.scalar('val accuracy grading', valid_accuracy_2.result(), step=epoch)

        print(f"Epoch {epoch + 1}/{max_epochs + 1}. Total validation loss {valid_loss.result():.5f}, "
              f"loss classification training: {val_loss_classi.result():.5f}, grading loss training: {val_loss_grading.result():.5f},"
              f"acc classification: {float(valid_accuracy_1.result()):.5f}, acc grading: {float(valid_accuracy_2.result()):.5f}, "
              f"f'F-1 classification: {val_metrics[2]}, F-1 avg: {np.mean(val_metrics[2]):.5f}, MSE grading: {val_metrics[3]}, "
              f"MSE avg: {np.mean(val_metrics[3]):.5f}")

        # checkpoint.save(epoch)
        # writer.flush()


        wait += 1
        if epoch == 0:
            best_loss = valid_loss.result()
        if valid_loss.result() < best_loss:
            best_loss = valid_loss.result()
            model.save_weights('best_model_weights.h5')
            wait = 0
        if wait >= patience:
            print('Early stopping triggered: wait time > patience')
            break

        if epoch % 5 == 0:
            df = pd.DataFrame(list(zip(epoch_counter, train_loss_list, train_classi_loss_list, train_grading_loss_list,
                                       val_loss_list, val_classi_loss_list, val_grading_loss_list,
                                       train_accuracy_list_1, train_accuracy_list_2,
                                       val_accuracy_list_1, val_accuracy_list_2,
                                       f1_train_classi_list, mse_train_grading_list,
                                       f1_val_classi_list, mse_val_grading_list
                                       )), columns=header_column)

            path_history_csv_file = os.path.join(results_directory, 'training_history.csv')
            df.to_csv(path_history_csv_file, index=False)

    model.save(filepath=model_dir, save_format='tf')
    print(f'model saved at {model_dir}')
    print('Total Training TIME:', (datetime.datetime.now() - start_time))

    # save history

    df = pd.DataFrame(list(zip(epoch_counter, train_loss_list, train_classi_loss_list, train_grading_loss_list,
                               val_loss_list, val_classi_loss_list, val_grading_loss_list,
                               train_accuracy_list_1, train_accuracy_list_2,
                               val_accuracy_list_1, val_accuracy_list_2,
                               f1_train_classi_list, mse_train_grading_list,
                               f1_val_classi_list, mse_val_grading_list
                               )), columns=header_column)

    path_history_csv_file = os.path.join(results_directory, 'training_history.csv')
    df.to_csv(path_history_csv_file, index=False)

    print(f'csv file with training history saved at: {path_history_csv_file}')

    if test_dataset:

        print(f'Making predictions on test dataset')
        # 2Do load saved model

        list_images = list()

        list_predictions_class_1 = list()
        list_predictions_class_2 = list()
        list_predictions_class_3 = list()
        list_predictions_class_4 = list()
        list_predictions_grade_1 = list()
        list_predictions_grade_2 = list()
        list_predictions_grade_3 = list()
        list_predictions_grade_4 = list()

        list_labels_class_1 = list()
        list_labels_class_2 = list()
        list_labels_class_3 = list()
        list_labels_class_4 = list()
        list_labels_grade_1 = list()
        list_labels_grade_2 = list()
        list_labels_grade_3 = list()
        list_labels_grade_4 = list()

        for j, data_batch in enumerate(tqdm.tqdm(test_dataset, desc='Making predictions on test dataset')):
            image_labels = data_batch[0]
            path_img = data_batch[1]
            image, labels = image_labels
            labels_classi, labels_grading = labels
            pred_classi, pred_grading = prediction_step(image)

            pred_class_1, pred_class_2, pred_class_3, pred_class_4 = pred_classi.numpy()[0]
            prediction_classi_1 = list(pred_class_1).index(max(list(pred_class_1)))
            prediction_classi_2 = list(pred_class_2).index(max(list(pred_class_2)))
            prediction_classi_3 = list(pred_class_3).index(max(list(pred_class_3)))
            prediction_classi_4 = list(pred_class_4).index(max(list(pred_class_4)))

            pred_grade_1, pred_grade_2, pred_grade_3, pred_grade_4 = pred_grading.numpy()[0]
            predictions_grade_1 = list(pred_grade_1).index(max(list(pred_grade_1)))
            predictions_grade_2 = list(pred_grade_2).index(max(list(pred_grade_2)))
            predictions_grade_3 = list(pred_grade_3).index(max(list(pred_grade_3)))
            predictions_grade_4 = list(pred_grade_4).index(max(list(pred_grade_4)))

            list_images.append(path_img.numpy())

            list_predictions_class_1.append(prediction_classi_1)
            list_predictions_class_2.append(prediction_classi_2)
            list_predictions_class_3.append(prediction_classi_3)
            list_predictions_class_4.append(prediction_classi_4)
            list_labels_class_1.append(labels_classi.numpy()[0][0])
            list_labels_class_2.append(labels_classi.numpy()[0][1])
            list_labels_class_3.append(labels_classi.numpy()[0][2])
            list_labels_class_4.append(labels_classi.numpy()[0][3])

            list_predictions_grade_1.append(predictions_grade_1)
            list_predictions_grade_2.append(predictions_grade_2)
            list_predictions_grade_3.append(predictions_grade_3)
            list_predictions_grade_4.append(predictions_grade_4)
            list_labels_grade_1.append(labels_grading.numpy()[0][0])
            list_labels_grade_2.append(labels_grading.numpy()[0][1])
            list_labels_grade_3.append(labels_grading.numpy()[0][2])
            list_labels_grade_4.append(labels_grading.numpy()[0][3])

        header_column = list()
        header_column.insert(0, 'img name')
        header_column.append('predicted overall')
        header_column.append('predicted bleeding')
        header_column.append('predicted MI')
        header_column.append('predicted TI')

        header_column.append('label overall')
        header_column.append('label bleeding')
        header_column.append('label MI')
        header_column.append('label TI')

        header_column.append('predicted grade overall')
        header_column.append('predicted grade bleeding')
        header_column.append('predicted grade MI')
        header_column.append('predicted grade TI')

        header_column.append('label grade overall')
        header_column.append('label grade bleeding')
        header_column.append('label grade MI')
        header_column.append('label grade TI')

        df = pd.DataFrame(list(zip(list_images, list_predictions_class_1, list_predictions_class_2,
                                   list_predictions_class_3, list_predictions_class_4,
                                   list_labels_class_1, list_labels_class_2,
                                   list_labels_class_3, list_labels_class_4,
                                   list_predictions_grade_1, list_predictions_grade_2,
                                   list_predictions_grade_3, list_predictions_grade_4,
                                   list_labels_grade_1, list_labels_grade_2,
                                   list_labels_grade_3, list_labels_grade_4)), columns=header_column)

        path_results_csv_file = os.path.join(results_directory, 'predictions.csv')
        df.to_csv(path_results_csv_file, index=False)

        print(f'csv file with results saved: {path_results_csv_file}')

        dir_conf_matrix_grade_nothing = os.path.join(results_directory, 'confusion_matrix_grade_nothing.png')
        daa.compute_confusion_matrix(list_labels_grade_1, list_predictions_grade_1, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_grade_nothing)
        dir_conf_matrix_grade_bleeding = os.path.join(results_directory, 'confusion_matrix_grade_bleeding.png')
        daa.compute_confusion_matrix(list_labels_grade_2, list_predictions_grade_2, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_grade_bleeding)
        dir_conf_matrix_grade_mechanical_injury = os.path.join(results_directory, 'confusion_matrix_grade_mechanical_injury.png')
        daa.compute_confusion_matrix(list_labels_grade_3, list_predictions_grade_3, plot_figure=False,
                                     dir_save_fig=dir_conf_matrix_grade_mechanical_injury)
        dir_conf_matrix_grade_thermal_injury = os.path.join(results_directory, 'confusion_matrix_grade_thermal_injury.png')
        daa.compute_confusion_matrix(list_labels_grade_4, list_predictions_grade_4, plot_figure=False,
                                     dir_save_fig=dir_conf_matrix_grade_thermal_injury)

        dir_conf_matrix_class_overall = os.path.join(results_directory, 'confusion_matrix_class_overall.png')
        daa.compute_confusion_matrix(list_labels_class_1, list_predictions_class_1, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_class_overall)
        dir_conf_matrix_class_bleeding = os.path.join(results_directory, 'confusion_matrix_class_bleeding.png')
        daa.compute_confusion_matrix(list_labels_class_2, list_predictions_class_2, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_class_bleeding)
        dir_conf_matrix_class_mechanical_injury = os.path.join(results_directory, 'confusion_matrix_class_mechanical_injury.png')
        daa.compute_confusion_matrix(list_labels_class_3, list_predictions_class_3, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_class_mechanical_injury)
        dir_conf_matrix_class_thermal_injury = os.path.join(results_directory, 'confusion_matrix_class_thermal_injury.png')
        daa.compute_confusion_matrix(list_labels_class_4, list_predictions_class_4, plot_figure=False,
                                 dir_save_fig=dir_conf_matrix_class_thermal_injury)





def main(_argv):
    institution_folders_frames = {'stras': 'stras_by70', 'bern': 'bern_by70'}
    institution_folders_annotations = {'stras': 'stras_70', 'bern': 'bern_70'}
    physical_devices = tf.config.list_physical_devices('GPU')
    print('Build with Cuda:', tf.test.is_built_with_cuda())
    print("Num GPUs:", len(physical_devices))
    ratio = 1
    path_dataset = FLAGS.path_dataset
    path_annotations_dataset = FLAGS.path_annotations
    data_center = FLAGS.data_center
    fold = FLAGS.fold
    output_type = FLAGS.output_type
    type_training = FLAGS.type_training
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    results_dir = FLAGS.results_dir
    learning_rate = FLAGS.learning_rate
    backbone_network = FLAGS.backbone
    train_verbose = FLAGS.train_verbose
    input_shape = FLAGS.input_shape
    image_size = FLAGS.image_size
    patch_size = FLAGS.patch_size
    projection_dim = FLAGS.projection_dim
    transformer_layers = FLAGS.transformer_layers
    num_heads = FLAGS.num_heads
    mlp_head_units = FLAGS.mlp_head_units
    num_classes = FLAGS.num_classes
    input_size_model = (input_shape, input_shape)

    selected_classes = ['Overall', 'Bleeding', 'Mechanical injury', 'Thermal injury']

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = {'classi': ['accuracy'], #tf.keras.metrics.Recall()], #, tf.keras.metrics.Recall()],
               'grading': ['accuracy'], #tf.keras.metrics.Recall()]#, tf.keras.metrics.Recall()],
               }
    train_backbone = FLAGS.train_backbone

    if data_center == 'both':
        train_dataset_dict = {}
        valid_dataset_dict = {}
        test_dataset_dict = {}
        for center in institution_folders_frames:
            path_frames = os.path.join(path_dataset, institution_folders_frames[center], 'frames')
            path_annotations = os.path.join(path_annotations_dataset, institution_folders_annotations[center])

            path_train_annotations = os.path.join(path_annotations, 'train')
            path_val_annotations = os.path.join(path_annotations, 'val')
            path_test_annotations = os.path.join(path_annotations, 'test')

            train_annotations_file_name = [f for f in os.listdir(path_train_annotations) if fold + '.pickle' in f][0]
            val_annotations_file_name = [f for f in os.listdir(path_val_annotations) if fold + '.pickle' in f][0]
            test_annotations_file_name = [f for f in os.listdir(path_test_annotations) if fold + '.pickle' in f][0]

            train_annotations_file_path = os.path.join(path_train_annotations, train_annotations_file_name)
            val_annotations_file_path = os.path.join(path_val_annotations, val_annotations_file_name)
            test_annotations_file_path = os.path.join(path_test_annotations, test_annotations_file_name)

            temp_train_dataset_dict = dam.load_dataset_from_directory(path_frames, train_annotations_file_path, output_type=output_type, ratio=ratio)
            temp_valid_dataset_dict = dam.load_dataset_from_directory(path_frames, val_annotations_file_path, output_type=output_type, ratio=ratio)
            temp_test_dataset_dict = dam.load_dataset_from_directory(path_frames, test_annotations_file_path, output_type=output_type, ratio=ratio)

            train_dataset_dict = {**train_dataset_dict, **temp_train_dataset_dict}
            valid_dataset_dict = {**valid_dataset_dict, **temp_valid_dataset_dict}
            test_dataset_dict = {**test_dataset_dict, **temp_test_dataset_dict}

    else:
        if data_center == 'stras':
            other_data_center = 'bern'
        elif data_center == 'bern':
            other_data_center = 'stras'

        path_frames = os.path.join(path_dataset, institution_folders_frames[data_center], 'frames')
        path_cross_center_frames = os.path.join(path_dataset, institution_folders_frames[other_data_center], 'frames')
        path_annotations = os.path.join(path_annotations_dataset, institution_folders_annotations[data_center])
        cross_center_annotations = os.path.join(path_annotations_dataset,
                                                institution_folders_annotations[other_data_center])

        path_train_annotations = os.path.join(path_annotations, 'train')
        path_val_annotations = os.path.join(path_annotations, 'val')
        path_test_annotations_1 = os.path.join(path_annotations, 'test')
        path_test_annotations_2 = os.path.join(cross_center_annotations, 'test')

        train_annotations_file_name = [f for f in os.listdir(path_train_annotations) if fold + '.pickle' in f][0]
        val_annotations_file_name = [f for f in os.listdir(path_val_annotations) if fold + '.pickle' in f][0]
        test_annotations_file_name_1 = [f for f in os.listdir(path_test_annotations_1) if fold + '.pickle' in f][0]
        test_annotations_file_name_2 = [f for f in os.listdir(path_test_annotations_2) if fold + '.pickle' in f][0]

        train_annotations_file_path = os.path.join(path_train_annotations, train_annotations_file_name)
        val_annotations_file_path = os.path.join(path_val_annotations, val_annotations_file_name)
        test_annotations_file_path_1 = os.path.join(path_test_annotations_1, test_annotations_file_name_1)
        test_annotations_file_path_2 = os.path.join(path_test_annotations_2, test_annotations_file_name_2)

        train_dataset_dict = dam.load_dataset_from_directory(path_frames, train_annotations_file_path, output_type=output_type, ratio=ratio)
        valid_dataset_dict = dam.load_dataset_from_directory(path_frames, val_annotations_file_path, output_type=output_type, ratio=ratio)
        test_dataset_dict_1 = dam.load_dataset_from_directory(path_frames, test_annotations_file_path_1, output_type=output_type, ratio=ratio)
        test_dataset_dict_2 = dam.load_dataset_from_directory(path_cross_center_frames, test_annotations_file_path_2, output_type=output_type, ratio=ratio)
        test_dataset_dict = {**test_dataset_dict_1, **test_dataset_dict_2}

    grade_keys = ['Overall', 'Bleeding grade', 'Mechanical injury grade', 'Thermal injury grade']
    train_dataset = dam.make_tf_image_dataset(train_dataset_dict, training_mode=True, selected_labels=selected_classes,
                                              input_size=input_size_model, batch_size=batch_size,
                                              multi_output_size=[4, len(grade_keys)],
                                              extract_label_by_keys=grade_keys)

    valid_dataset = dam.make_tf_image_dataset(valid_dataset_dict, training_mode=False, selected_labels=selected_classes,
                                              input_size=input_size_model, batch_size=batch_size,
                                              multi_output_size=[4, len(grade_keys)],
                                              extract_label_by_keys=grade_keys)

    test_dataset = dam.make_tf_image_dataset(test_dataset_dict, training_mode=False, selected_labels=selected_classes,
                                             input_size=input_size_model, batch_size=max([1, len(physical_devices)]),
                                             multi_output_size=[4, len(grade_keys)],
                                             image_paths=True,
                                             extract_label_by_keys=grade_keys)

    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim, ]  # Size of the transformer layers
    mlp_head_u = [int(x) for x in mlp_head_units]
    model_input_shape = (input_shape, input_shape, 3)
    num_classes = [4, 4]
    if type_training == 'custom_training':

        custom_training(name_model, train_dataset, valid_dataset, epochs, fold=fold,
                  batch_size=batch_size, learning_rate=0.0001, results_dir=results_dir, metrics=metrics, optimizer=optimizer,
                  test_dataset=test_dataset, output_type=output_type,
                  input_shape=model_input_shape, image_size=image_size, patch_size=patch_size, num_patches=num_patches,
                  projection_dim=projection_dim, transformer_layers=transformer_layers, num_heads=num_heads,
                  transformer_units=transformer_units, mlp_head_units=mlp_head_u, num_classes=num_classes,
                        verbose=train_verbose, gpus_available=len(physical_devices))

    elif type_training == 'fit_model':
        model_fit(name_model, train_dataset, valid_dataset, epochs, fold=fold,
                  batch_size=batch_size, learning_rate=0.0001, results_dir=results_dir, metrics=metrics, optimizer=optimizer,
                  test_dataset=test_dataset, output_type=output_type,
                  input_shape=model_input_shape, image_size=image_size, patch_size=patch_size, num_patches=num_patches,
                  projection_dim=projection_dim, transformer_layers=transformer_layers, num_heads=num_heads,
                  transformer_units=transformer_units, mlp_head_units=mlp_head_u, num_classes=num_classes
                  )

    else:
        print(f'{type_training} not in options!')


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_list('selected_classes', ['Overall', 'Bleeding', 'Mechanical injury', 'Thermal injury'], 'classes selected')
    flags.DEFINE_string('path_dataset', '', 'directory dataset')
    flags.DEFINE_string('path_annotations', '', 'directory annotations')
    flags.DEFINE_string('data_center', 'both', 'which sub-division to use [stras, bern] or both')
    flags.DEFINE_string('fold', '1', 'fold od the dataset')
    flags.DEFINE_string('output_type', 'binary', 'binary or level')
    flags.DEFINE_integer('clips_size', 5, 'number of clips')
    flags.DEFINE_string('type_training', 'custom_training', 'eager_train or custom_training')
    flags.DEFINE_integer('batch_size',8,'batch size')
    flags.DEFINE_integer('epochs', 1, 'epochs')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('analyze_data', True, 'analyze the data after the experiment')
    flags.DEFINE_string('backbone', 'resnet50', 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')
    flags.DEFINE_string('pretrained_weights', '','pretrained weights for the backbone either [''(none), "imagenet", "path_to_weights"]')
    flags.DEFINE_boolean('train_backbone', False, 'train the backbone')
    flags.DEFINE_boolean('train_verbose', False, 'show training evolution per batch')

    flags.DEFINE_integer('patch_size', 16, 'size of the patch')
    flags.DEFINE_integer('input_shape', 224, 'input shape')
    flags.DEFINE_integer('image_size', 224, 'size of the input image') # or 72?
    flags.DEFINE_integer('projection_dim', 256, 'projections dim') # or 64, 768?
    flags.DEFINE_integer('transformer_layers', 8, 'num of layers in the transformer')
    flags.DEFINE_integer('num_heads', 4, 'number of heads')
    flags.DEFINE_list('mlp_head_units', [256, 128], 'number of mlp head units')
    flags.DEFINE_integer('num_classes', 4, 'number of classes')

    try:
        app.run(main)
    except SystemExit:
        pass