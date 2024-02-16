import torch
import random
import numpy as np
import os
import pickle
import tqdm
import copy
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import gc
from IPython import display
import pylab as pl
from tensorflow.keras.layers import *
from sklearn.manifold import TSNE
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import datetime
from models import classification_models as cm


input_sizes_models = {'resnet50': (224, 224), 'mobilenet': (224, 224),
                      'densenet121': (224, 224)}


def get_preprocess_input_backbone(name_backbone, x):
    if name_backbone == 'resnet50':
        preprocess_input = tf.keras.applications.resnet50.preprocess_input(x)
    elif name_backbone == 'densenet121':
        preprocess_input = tf.keras.applications.densenet.preprocess_input(x)
    elif name_backbone == 'mobilenet':
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input(x)
    else:
        raise ValueError(f'Preprocess input {name_backbone} not avialable')
    return preprocess_input


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False):
    if name_model == 'resnet50':
        base_model = tf.keras.applications.resnet50.ResNet50(include_top=include_top, weights=weights, pooling='avg')
        base_model.trainable = True

    elif name_model == 'mobilenet':
        base_model = tf.keras.applications.MobileNet(include_top=include_top, weights=weights, pooling='avg')
        base_model.trainable = trainable

    else:
        raise ValueError(f' MODEL: {name_model} not found')

    return base_model


def build_pretrained_model(name_model, trainable=False):
    input_image = tf.keras.Input(shape=(128, 128, 3), name="image")
    x = tf.image.resize(input_image, input_sizes_models[name_model], method='area')
    x = get_preprocess_input_backbone(name_model, x)
    #base_model = load_pretrained_backbones(name_model)
    base_model = cm.load_pretrained_backbones_from_local(name_model)
    if trainable is False:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model(x)
    output_layer = x
    print('output layer', output_layer.shape)
    image_encoder = tf.keras.Model(inputs=input_image, outputs=output_layer, name=f'pretrained_model_{name_model}')
    # input_transformer_shape = (output_layer.shape[1], output_layer.shape[2])
    return image_encoder


def get_bleeding_level(dict_frame):
    bleeding_level = 0
    if dict_frame['Bleeding - 1'] == 1:
        bleeding_level = 1
    elif dict_frame['Bleeding - 2'] == 1:
        bleeding_level = 1
    elif dict_frame['Bleeding - 3'] == 1:
        bleeding_level = 2
    elif dict_frame['Bleeding - 4'] == 1:
        bleeding_level = 2
    elif dict_frame['Bleeding - 5'] == 1:
        bleeding_level = 2

    return bleeding_level

def get_mi_level(dict_frame):
    mi_level = 0
    if dict_frame['Mechanical injury - 1'] == 1:
        mi_level = 1
    elif dict_frame['Mechanical injury - 2'] == 1:
        mi_level = 1
    elif dict_frame['Mechanical injury - 3'] == 1:
        mi_level = 2
    elif dict_frame['Mechanical injury - 4'] == 1:
        mi_level = 2
    elif dict_frame['Mechanical injury - 5'] == 1:
        mi_level = 2
    return mi_level

def get_ti_level(dict_frame):
    ti_level = 0
    if dict_frame['Thermal injury - 1'] == 1:
        ti_level = 1
    elif dict_frame['Thermal injury - 2'] == 1:
        ti_level = 1
    elif dict_frame['Thermal injury - 3'] == 1:
        ti_level = 2
    elif dict_frame['Thermal injury - 4'] == 1:
        ti_level = 2
    elif dict_frame['Thermal injury - 5'] == 1:
        ti_level = 2
    return ti_level


def get_labels_class(dataset_dict):
    Mechanical_injury = dataset_dict.get('Mechanical injury')
    Thermal_injury = dataset_dict.get('Thermal injury')
    Overall = dataset_dict.get('Overall')
    Bleeding = dataset_dict.get('Bleeding')
    #labels_class = [Overall, Bleeding, Mechanical_injury, Thermal_injury]
    labels_class = [Bleeding, Mechanical_injury, Thermal_injury]
    out_labels = [x if x != None else 0 for x in labels_class]
    return out_labels

def get_labels_grade(dataset_dict):
    Overall_grade = dataset_dict.get('Overall')
    Bleeding_grade = dataset_dict.get('Bleeding grade')
    Thermal_injury_grade = dataset_dict.get('Thermal injury grade')
    Mechanical_injury_grade = dataset_dict.get('Mechanical injury grade')
    labels_grade = [Bleeding_grade, Mechanical_injury_grade, Thermal_injury_grade]
    out_labels = [x if x != None else 0 for x in labels_grade]
    return out_labels


def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = tf.keras.Input(shape=(10,64,64,3))
	x = Reshape((64,64,30))(inputs)
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)
	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	model = tf.keras.Model(inputs, outputs)
	# return the model to the calling function
	return model

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.keras.activations.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def summary(model):

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    return trainable_count+non_trainable_count


def load_dataset_from_directory(path_frames, path_annotations, output_type='binary',
                                class_condition='', num_samples=None, ratio=None):
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
    print(f'Reading dir:{path_annotations}')
    for case in tqdm.tqdm(list_cases, desc=f"Loading data"):
        annotated_frames = pickleInfo[case]
        frames_IDs = [d['Frame_id'] for d in annotated_frames]
        path_case = os.path.join(path_frames, case)
        list_imgs = os.listdir(path_case)
        # double-check the list since not all the frames have annotations
        list_imgs = [f.split('.')[0] for f in list_imgs if f.split('.')[0] in frames_IDs]
        list_real_imgs = [f for f in annotated_frames if
                          os.path.isfile(os.path.join(path_frames, case, f['Frame_id'] + '.jpg'))]

        if output_type == 'binary':
            dict_frames = {case + d['Frame_id']: {'case_id': case,
                                                  'Frame_id': d['Frame_id'],
                                                  'Path_img': os.path.join(path_frames, case, d['Frame_id'] + '.jpg'),
                                                  'Phase_gt': d['Phase_gt'],
                                                  'Step_gt': d['Step_gt'],
                                                  'Overall': d['Overall'],
                                                  'Bleeding': d['Bleeding'],
                                                  'Mechanical injury': d['Mechanical injury'],
                                                  'Thermal injury': d['Thermal injury'],
                                                  'Event_ID': d['Event_ID']
                                                  } for d in annotated_frames if d['Frame_id'] in list_imgs}
        elif output_type == 'level':

            dict_frames = {case + d['Frame_id']: {'case_id': case,
                                                  'Frame_id': d['Frame_id'],
                                                  'Path_img': os.path.join(path_frames, case, d['Frame_id'] + '.jpg'),
                                                  'Phase_gt': d['Phase_gt'],
                                                  'Step_gt': d['Step_gt'],
                                                  'Overall': d['Overall'],
                                                  'Bleeding': get_bleeding_level(d),
                                                  'Mechanical injury': get_mi_level(d),
                                                  'Thermal injury': get_ti_level(d),
                                                  'Event_ID': d['Event_ID']
                                                  } for d in annotated_frames if d['Frame_id'] in list_imgs}

        elif output_type == 'class_and_grade':

            dict_frames = {case + d['Frame_id']: {'case_id': case,
                                                  'Frame_id': d['Frame_id'],
                                                  'Path_img': os.path.join(path_frames, case, d['Frame_id'] + '.jpg'),
                                                  'Phase_gt': d['Phase_gt'],
                                                  'Step_gt': d['Step_gt'],
                                                  'Overall': d['Overall'],
                                                  'Bleeding': d['Bleeding'],
                                                  'Mechanical injury': d['Mechanical injury'],
                                                  'Thermal injury': d['Thermal injury'],
                                                  'Bleeding grade': get_bleeding_level(d),
                                                  'Mechanical injury grade': get_mi_level(d),
                                                  'Thermal injury grade': get_ti_level(d),
                                                  'Event_ID': d['Event_ID']
                                                  } for d in annotated_frames if d['Frame_id'] in list_imgs}

        output_dict = {**output_dict, **dict_frames}
        # option b) with ChainMap, check which one is more efficient
        # output_dict = dict(ChainMap({}, output_dict, dict_frames))

    if num_samples:
        new_output_dict = {}
        keys_dict = list(output_dict.keys())
        total_samples = 0
        temp_dict = {}
        if ratio:
            num_neg = 0
            num_pos = 0
            total_neg = int(num_samples*ratio)
            total_pos = num_samples - total_neg
            while num_pos + num_neg <= num_samples:
                r = random.choice(keys_dict)

                if output_dict[r]['Overall'] == 1 and num_pos <= total_pos:
                    temp_dict[r] = output_dict[r]
                    num_pos += 1
                elif output_dict[r]['Overall'] == 0 and num_neg <= total_neg:
                    temp_dict[r] = output_dict[r]
                    num_neg += 1

            new_output_dict = {**new_output_dict, **temp_dict}
        else:

            while total_samples <= num_samples:
                r = random.choice(keys_dict)
                temp_dict[r] = output_dict[r]
                new_output_dict = {**new_output_dict, **output_dict[r]}
                total_samples += 1

        output_dict = copy.copy(new_output_dict)

    if ratio:
        new_output_dict = {}
        total_neg = 0
        keys_dict = list(output_dict.keys())
        total_frames = len(keys_dict)

        for k in keys_dict:
            if output_dict[k]['Bleeding'] == 1 or output_dict[k]['Mechanical injury'] == 1 or output_dict[k]['Thermal injury'] == 1:
                total_neg += 1

        total_pos = int(total_neg * ratio)
        prob = (total_pos/(total_frames-total_neg))

        for k in keys_dict:
            if output_dict[k]['Bleeding'] == 1 or output_dict[k]['Mechanical injury'] == 1 or output_dict[k]['Thermal injury'] == 1:
                #temp_dict = output_dict[k]
                new_output_dict[k] = output_dict[k]

            else:
                if random.random() <= prob:
                    #temp_dict = output_dict[k]
                    new_output_dict[k] = output_dict[k]

        output_dict = copy.copy(new_output_dict)

    print(f'Dataset with {len(output_dict)} elements')
    return output_dict


def transformer(n_frames=10):
    proj_dim = 128
    transformer_layers = 2
    projection_dim = proj_dim
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    mlp_head_units = [
        256,
        128,
    ]
    input_transformer_shape = (n_frames,1024)
    inputs_image = Input(shape=input_transformer_shape)
    inputs_position = Input(shape = (input_transformer_shape[0]))

    s1 = np.sqrt(input_transformer_shape)[0].astype(np.int16)
    image_projection = Dense(projection_dim)(inputs_image)
    position_projection = Embedding(67,projection_dim)(inputs_position)
    encoded_patches = Add()([image_projection, position_projection])
    for _ in range(transformer_layers):
        # Layer normalization 1.
        # x1 = encoded_patches
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        # x3 = x2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    op = GlobalAveragePooling1D()(encoded_patches)
    op = Dense(96, activation = 'tanh')(op)
    transformer1 = tf.keras.Model((inputs_image, inputs_position), op)

    return transformer1


def make_featureExtractor(n_frames=10):
    input_transformer_shape = (n_frames, 1024)
    transformer1 = transformer(n_frames)
    inputs_image = Input(shape=(n_frames, 128, 128, 3))
    inputs_position = Input(shape = (input_transformer_shape[0]))
    #m_met.trainable = False
    #conv_2d_layer = m_met
    conv_2d_layer = build_pretrained_model('mobilenet')
    outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs_image)
    emb = transformer1((outputs, inputs_position))
    return tf.keras.Model((inputs_image, inputs_position), emb)

def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	labels = [str(i) for i in labels]
	unique_label = np.unique(labels)
	labels_dict = {unique_label[i]:i for i in range(len(unique_label))}
	labels = np.asarray([labels_dict.get(i) for i in labels])
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]
		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))


def train(i, f_m, label_op, img_np, n_frames=10):
    input_transformer_shape = (n_frames, 1024)
    l1 = label_op[:,i]
    pI, pL = make_pairs(img_np, l1)
    positions = np.asarray([[j for j in range(input_transformer_shape[0])] for _ in range(len(pI))])
    i1 = pI[:,0,:,:,:]
    i2 = pI[:,1,:,:,:]
    for _ in range(5):
        f_m.fit((i1, i2, positions), pL, verbose=1, batch_size=16)
        gc.collect()


def create_val_plots(valid_generator, featureExtractors_dict, gg, dataset, max_iter=20, path_results=os.getcwd(), n_frames=10):
    input_transformer_shape = (n_frames, 1024)
    idx_dict = {'bleeding': 0, 'mi': 1, 'ti': 2}
    for k in featureExtractors_dict.keys():
        featureExtractor = featureExtractors_dict.get(k)
        e = []
        l = []
        j = 0
        for path, label, label_op, img_np in valid_generator:
            gc.collect()
            path, label, label_op, img_np = path, label, label_op.numpy(), img_np.numpy()
            positions = np.asarray([[i for i in range(input_transformer_shape[0])] for _ in range(len(img_np))])
            emb = featureExtractor.predict((img_np, positions), batch_size=16, verbose=1)
            j += 1
            e.append(emb)
            l.append(label_op)
            if j >= max_iter:
                break

        e = np.asarray(e)
        l = np.asarray(l)
        e = np.reshape(e, (-1, 96))
        l = np.reshape(l, (-1, 3))

        for ind in idx_dict:
            idxx = idx_dict[ind]
            label1 = []
            uq = {}
            h = 0
            for i in l:
                # i = i[0]
                i = i[idxx]
                if str(i) not in uq:
                    uq[str(i)] = h
                    h += 1
                label1.append(uq.get(str(i)))

            emb = e
            p = len(emb) // len(uq)
            # p = 32
            if p >= len(label1):
                p = 32
            model = TSNE(n_components=2, random_state=0, perplexity=p)
            op_emb = model.fit_transform(emb)
            # imhandle = plt.scatter(op_emb[:,0], op_emb[:,1], s = 3, c=label1, cmap = 'jet')
            plt.figure(idxx + 3, figsize=(12, 8))

            fig, ax = plt.subplots()
            plt.title(f'TSNE {k} iter: {str(gg)}')
            scatter = ax.scatter(op_emb[:, 0], op_emb[:, 1], s=3, c=label1, cmap='jet')
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="best", title="Grades")
            ax.add_artist(legend1)
            # produce a legend with a cross-section of sizes from the scatter
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
            path_plot_val = os.path.join(path_results, f'tsne_plot_{k}_{ind}_{dataset}.png')
            plt.savefig(path_plot_val)
            plt.clf()
            plt.close()




def create_model(n_frames=10):
    input_transformer_shape = (n_frames, 1024)
    imgA = tf.keras.Input(shape=(n_frames, 128, 128, 3))
    imgB = tf.keras.Input(shape=(n_frames, 128, 128, 3))
    inputs_position = tf.keras.Input(shape=(input_transformer_shape[0]))

    featureExtractor1 = make_featureExtractor(n_frames)
    # featureExtractor = build_siamese_model((10,64,64,3))
    featsA = featureExtractor1((imgA, inputs_position))
    featsB = featureExtractor1((imgB, inputs_position))

    # finally, construct the siamese network
    distance = Lambda(euclidean_distance)([featsA, featsB])
    final_model = tf.keras.Model(inputs=[imgA, imgB, inputs_position], outputs=distance)
    final_model.compile(loss=contrastive_loss, optimizer=tf.keras.optimizers.Adam(1e-5))
    return final_model, featureExtractor1


class Dataset_v1(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dictionary_labels, min_num_events=50, training=False, size=64, rotate90=False, random_flip=False,
                 tubelet_size=5):
        self.size = size
        self.dictionary_labels = dictionary_labels
        self.training = training
        self.min_num_events = min_num_events
        self.rotate90 = rotate90
        self.random_flip = random_flip
        self.tubelet_size = tubelet_size

        self.list_files = list(dictionary_labels.keys())
        self.unique_labels = []
        self.label_path_dict = {}
        self.strLabel_to_nparr = {}

        for img_id in self.list_files:
            img_path = self.dictionary_labels.get(img_id).get('Path_img')
            # img_label = np.asarray([get_labels_class(self.dictionary_labels[img_id]),
            #                        get_labels_grade(self.dictionary_labels[img_id])])
            img_label = np.asarray(get_labels_class(self.dictionary_labels[img_id]))
            self.unique_labels.append(str(img_label))
            if str(img_label) not in self.strLabel_to_nparr:
                self.strLabel_to_nparr[str(img_label)] = img_label

            if str(img_label) not in self.label_path_dict:
                self.label_path_dict[str(img_label)] = []
                self.label_path_dict.get(str(img_label)).append(img_path)
            else:
                self.label_path_dict.get(str(img_label)).append(img_path)

        self.uq, counts = np.unique(self.unique_labels, return_counts=True)
        # self.pairs = list(zip(list_all_events, list_all_labels))

        self.label_list = [self.uq[i] for i in range(len(self.uq)) if counts[i] > 50]
        #   random.choice(label_path_dict.get(random.choice(label_list)))
        self.net_paths = sum([self.label_path_dict.get(k) for k in self.label_path_dict.keys()], [])
        self.list_IDs = [i for i in range(len(self.net_paths))]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        ID = self.list_IDs[index]
        # Select sample
        # op_img = np.zeros((self.tubelet_size, self.size, self.size, 3))
        op_img = []
        label = random.choice(self.label_list)
        label_op = self.strLabel_to_nparr.get(label)

        path = random.choice(self.label_path_dict.get(label))
        for i in range(self.tubelet_size):
            try:
                img_temp = Image.open(path)
                img_resized = img_temp.resize((self.size, self.size))
                img_np = np.asarray(img_resized) / 255.0
                imf_final = img_np
            except:
                imf_final = np.zeros((self.size, self.size, 3))

            op_img.append(imf_final)

            prev_frame = int(path.split('.')[0].split('/')[-1]) - 1
            prev_frame = '/' + '{0:08d}'.format(prev_frame) + '.jpg'
            path = ('/').join(path.split('.')[0].split('/')[:-1]) + prev_frame

        # img_temp = Image.open(path)
        # img_resized = img_temp.resize((self.size, self.size))
        # img_np = np.asarray(img_resized)/255.0
        # imf_final = img_np
        if self.training is True:
            # Augmentations
            y = np.random.uniform(0.4, 2.0)
            imf_final = np.asarray(op_img) * y
            imf_final = np.ascontiguousarray(np.rot90(imf_final, np.random.randint(0, 4), axes=(1, 2)))
        else:
            imf_final = np.asarray(op_img)

        return path, label, label_op, imf_final  # , img_final

    def augment(self, img):
        out_img = img
        if self.rotate90 is True:
            state_rot = random.randint(0, 2)
            if state_rot == 1:
                out_img = np.rot90(out_img, k=random.randint(0, 4))
        if self.random_flip is True:
            state_flip = random.randint(0, 2)
            if state_flip == 1:
                out_img = np.fliplr(out_img)
        return out_img

#def run_metrics(y_true, y_pred):
#    f1 = [f1_score(y_true[:,i], y_pred[:,i], average='weighted') for i in range(y_pred.shape[1])]
#    #aps = [average_precision_score(y_true[:,i], y_pred[:,i], average='weighted') for i in range(y_pred.shape[1])]
#    mse = [mean_squared_error(y_true[:,i], y_pred[:,i]) for i in range(y_pred.shape[1])]
#    return f1, mse

def main(_argv):
    physical_devices = tf.config.list_physical_devices('GPU')
    print('Build with Cuda:', tf.test.is_built_with_cuda())
    print("Num GPUs:", len(physical_devices))
    tf.keras.backend.clear_session()
    path_pickle = FLAGS.path_pickle_train
    path_dataset = FLAGS.path_dataset
    institution_folders_frames = {'stras': 'stras_by70', 'bern': 'bern_by70'}
    center = FLAGS.data_center
    name_model = FLAGS.name_model
    plot_name = FLAGS.plot_name
    backbone_net = FLAGS.backbone_net
    training_date_time = datetime.datetime.now()
    experiment_name = ''.join(['contrastive_exp_', backbone_net, '_', training_date_time.strftime("%d_%m_%Y_%H_%M")])
    path_results = os.path.join(os.getcwd(), 'results', experiment_name)
    os.mkdir(path_results)
    n_frames = FLAGS.num_frames_input
    print(f'path experiment results {path_results}')

    path_frames = os.path.join(path_dataset, institution_folders_frames[center], 'frames')
    dataset_dict = load_dataset_from_directory(path_frames, path_pickle, output_type='class_and_grade')
    path_pickle_val = FLAGS.path_pickle_val
    dataset_dict_val = load_dataset_from_directory(path_frames, path_pickle_val, output_type='class_and_grade')

    d2 = Dataset_v1(dictionary_labels=dataset_dict_val, rotate90=True, random_flip=True, tubelet_size=n_frames, size=128)
    d1 = Dataset_v1(dictionary_labels=dataset_dict, rotate90=True, random_flip=True, tubelet_size=n_frames, size=128, training=True)
    input_transformer_shape = (n_frames, 1024)
    params = {'batch_size': 128,
              'num_workers': 0}

    params_val = {'batch_size': 64,
                  'num_workers': 0}
    training_generator = torch.utils.data.DataLoader(d1, **params)
    valid_generator = torch.utils.data.DataLoader(d2, **params_val)

    model_dict = {'class_model': None}
    featureExtractors_dict = {'class_model': None}

    for k in model_dict:
        f_model, feature_extractor = create_model(n_frames)
        model_dict[k] = f_model
        featureExtractors_dict[k] = feature_extractor

    gg = 0
    for j in range(30):
        for path, label, label_op, img_np in training_generator:
            gc.collect()
            path, label, label_op, img_np = path, label, label_op.numpy(), img_np.numpy()
            # label_op = np.reshape(label_op, (len(label_op), -1))
            # label_op = label_op[:,1,1:]
            for idx, k in enumerate(model_dict.keys()):
                f_m = model_dict.get(k)
                train(idx, f_m, label_op, img_np, n_frames=n_frames)

            if gg % 5 == 0:
                # save the model
                for idxx, k in enumerate(featureExtractors_dict.keys()):
                    f_a = featureExtractors_dict.get(k)
                    name_model = os.path.join(path_results, ''.join([k, '_feature_extractor_.h5']))
                    f_a.save(name_model, save_format='h5')
                    gc.collect()
                    tf.keras.backend.clear_session()

            if gg % 20 == 0:
                create_val_plots(training_generator, featureExtractors_dict, gg, 'train', max_iter=10, n_frames=n_frames, path_results=path_results)
                gc.collect()

            # validation dataset
            if gg % 10 == 0:
                create_val_plots(valid_generator, featureExtractors_dict, gg, 'val', n_frames=n_frames, path_results=path_results)
            gg += 1
            if gg == 100:
                break


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'directory dataset')
    flags.DEFINE_string('path_pickle_train', '', 'directory annotations')
    flags.DEFINE_string('path_pickle_val', '', 'directory annotations val')
    flags.DEFINE_string('path_dataset', '', 'path dataset')
    flags.DEFINE_integer('num_frames_input', 10, 'size of the video clipsre')

    flags.DEFINE_string('data_center', 'both', 'which sub-division to use [stras, bern] or both')
    flags.DEFINE_string('plot_name', 'binary', 'binary or level')
    flags.DEFINE_string('backbone_net',     'mobilenet', 'backbone')
    try:
        app.run(main)
    except SystemExit:
        pass