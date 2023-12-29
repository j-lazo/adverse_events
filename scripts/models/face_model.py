import tensorflow as tf
import numpy as np
from utils.contrast_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss
import datetime
import os
from sklearn.manifold import TSNE
from utils import tsne_functions
from src.params import Params
from progressbar import *
from progressbar import ProgressBar
class face_model(tf.keras.Model):

    def __init__(self, params, model_name):
        super(face_model, self).__init__()
        img_size = (params.image_size, params.image_size, 3)
        if model_name == 'inception':
            self.base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=img_size)
        elif model_name == 'resnet50':
            self.base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=img_size)
        self.base_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1024 = tf.keras.layers.Dense(1024)
        self.dense512 = tf.keras.layers.Dense(512)
        self.dense256 = tf.keras.layers.Dense(256)
        self.dense127 = tf.keras.layers.Dense(128)
        self.embedding_layer = tf.keras.layers.Dense(units=params.embedding_size)

    def call(self, images):
        x = self.base_model(images)
        x = self.flatten(x)
        x = self.dense1024(x)
        x = self.dense512(x)
        x = self.dense256(x)
        x = self.embedding_layer(x)
        return x


class Trainer():

    def __init__(self, json_path, train_dataset, valid_dataset, ckpt_dir, log_dir, restore, name_base_model,
                 test=False, tsne_analysis=False, test_dataset=None, visualize_dataset=None):

        self.params = Params(json_path)
        self.name_model = name_base_model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.test = test
        self.model = face_model(self.params, self.name_model)
        self.tsne_analysis = tsne_analysis
        self.visualize_dataset = visualize_dataset

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.params.learning_rate,
                                                                          decay_steps=10000, decay_rate=0.96,
                                                                          staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer,
                                              train_steps=tf.Variable(0, dtype=tf.int64),
                                              valid_steps=tf.Variable(0, dtype=tf.int64),
                                              epoch=tf.Variable(0, dtype=tf.int64))
        self.ckptmanager = tf.train.CheckpointManager(self.checkpoint, ckpt_dir, 5)

        self.params.triplet_strategy = 'batch_hard'
        if self.params.triplet_strategy == "batch_all":
            self.loss = batch_all_triplet_loss

        elif self.params.triplet_strategy == "batch_hard":
            self.loss = batch_hard_triplet_loss

        elif self.params.triplet_strategy == "batch_adaptive":
            self.loss = adapted_triplet_loss

        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        log_dir += current_time + '/train/'
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)

        if restore == '1':
            print(f'restoring Checkpoint from:{self.ckptmanager.latest_checkpoint}')
            self.checkpoint.restore(self.ckptmanager.latest_checkpoint)
            print(f'Restored from Checkpoint : {self.ckptmanager.latest_checkpoint}')

        else:
            print('\nIntializing from scratch\n')


    #def __call__(self, epoch):

    #    for i in range(epoch):
    #        self.train(i, epoch)
    #        if self.valid:
    #            self.validate(i)

    def train(self, epoch, epochs):
        total_loss = 0
        widgets = [f'Train epoch {epoch}/{epochs} :', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.train_samples // self.params.batch_size) + 20).start()

        for i, (images, labels) in pbar(enumerate(self.train_dataset)):
            loss = self.train_step(images, labels)
            total_loss += loss

            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_step_loss', loss, step=self.checkpoint.train_steps)
            self.checkpoint.train_steps.assign_add(1)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('train_batch_loss', total_loss, step=epoch)

        self.checkpoint.epoch.assign_add(1)
        print('Train Loss over epoch {}: {}'.format(epoch, total_loss))

        if int(self.checkpoint.epoch) % 5 == 0:
            if self.visualize_dataset:
                save_path = self.ckptmanager.save()
                print(f'Saved Checkpoint for step {self.checkpoint.epoch.numpy()} : {save_path}\n')


    def visualize_tsne_map(self):

        features = None
        labels_vis = list()
        for x, (images, labels) in enumerate(self.visualize_dataset):
            embeddings = self.model(images)
            current_features = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10).numpy()

            if features is not None:
                features = np.concatenate((features, current_features))
                labels_vis.append(labels.numpy()[0])
            else:
                features = current_features
        if self.tsne_analysis:
            tsne = TSNE(n_components=2).fit_transform(features)
            vis_labels = [str(l) for l in labels_vis]
            tsne_functions.visualize_tsne(tsne, vis_labels)

    def validate(self, epoch, epochs):
        widgets = [f'Valid epoch {epoch}/{epochs}  :', Percentage(), ' ', Bar('='), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.valid_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in pbar(enumerate(self.valid_dataset)):
            loss = self.valid_step(images, labels)
            total_loss += loss

            with self.train_summary_writer.as_default():
                tf.summary.scalar('valid_step_loss', loss, step=self.checkpoint.valid_steps)
            self.checkpoint.valid_steps.assign_add(1)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('valid_batch_loss', total_loss, step=epoch)

        print('Validation Loss over epoch {}: {}'.format(epoch, total_loss))
        #if (epoch + 1) % 5 == 0:
        #    print('\nValidation Loss over epoch {}: {}\n'.format(epoch, total_loss))

    def test(self):
        widgets = ['Testing step', Percentage(), ' ', Bar('='), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.test_samples // self.params.batch_size) + 50).start()
        total_loss = 0

        for i, (images, labels) in pbar(enumerate(self.test_dataset)):
            loss = self.test_step(images, labels)
            total_loss += loss

        print(f'Validation Loss: {total_loss}\n')

    def intermediate_step(self, images):
        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        return embeddings

    @tf.function
    def train_step(self, images, labels):

        with tf.GradientTape() as tape:
            embeddings = self.model(images)
            embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
            loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    @tf.function
    def valid_step(self, images, labels):

        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)

        return loss

    @tf.function
    def test_step(self, images, labels):

        embeddings = self.model(images)
        embeddings = tf.math.l2_normalize(embeddings, axis=1, epsilon=1e-10)
        loss = self.loss(labels, embeddings, self.params.margin, self.params.squared)
        # 2Do: plot or save images
        return loss