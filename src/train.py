import argparse
import json
import pandas as pd
import os
import time
import random
import pescador
import numpy as np
# import tensorflow as tf
import models_baselines
import config_file, shared
import pickle as pk
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# TODO:
# - add preprocessing of the data `np.log(blabla)`
# - add regularization parameter
# - add validation loop

class ImageDataset(Dataset):
    def __init__(self, data_folder, file_index, file_ground_truth, config, transform=None, target_transform=None):
        self.config = config
        self.data_folder = data_folder
        self.ids, self.id2gt = shared.load_id2gt(file_ground_truth) # `ids` are `str`
        # 3 cols: index | freq-time repr as a `.pk` | mp3
        self.index = pd.read_csv(file_index, header=None, sep='\t') # contains all the data
        # Only keep those relevant to the dataset
        ids_int = [int(id) for id in self.ids]
        self.index = self.index.loc[self.index[0].isin(ids_int)]
        # we get indices from 0 to n, important for `id_torch` in `__getitem__`
        self.index = self.index.reset_index(drop=True) 
        # Potential transformations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, id_torch):
        # WARNING: `id_torch` is handled by `torch` to iterate through the dataset. 
        # it is different from the `id` of the files contained in `index.tsv`.
        id = self.index.loc[id_torch, 0]
        # convert `img_path` to a `str`
        img_path = self.index.loc[id_torch, 1]
        img_path = os.path.join(self.data_folder, img_path)
        # load from `.pk` file
        img_file = open(img_path, 'rb')
        image = pk.load(img_file)
        image = image.astype(np.float32) # `preprocess_librosa.py` saves as np.float16.
        image = np.expand_dims(image, 0)
        if self.config['pre_processing'] == 'logEPS':
            image = np.log10(image + np.finfo(float).eps)
        elif self.config['pre_processing'] == 'logC':
            image = np.log10(10000 * image + 1)
        # TODO: change to np.float32 if performance is bad.
        label = torch.Tensor(self.id2gt[str(id)])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_size = y.shape[0]
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__ == '__main__':
    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_train[args.configuration]

    # load config parameters used in 'preprocess_librosa.py',
    config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # set patch parameters
    if config['audio_rep']['type'] == 'waveform':
        raise ValueError('Waveform-based training is not implemented')

    elif config['audio_rep']['spectrogram_type'] == 'mel':
        config['xInput'] = config['n_frames']
        config['yInput'] = config['audio_rep']['n_mels']

    # load audio representation paths
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # load training data
    file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)

    # load validation data
    file_ground_truth_val = config_file.DATA_FOLDER + config['gt_val']
    [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)

    # set output
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    print('# Train:', len(ids_train))
    print('# Val:', len(ids_val))
    print('# Classes:', config['classes_vector'])

    # save experimental settings
    experiment_id = str(shared.get_epoch_time()) + args.configuration
    model_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    json.dump(config, open(model_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    print('\nCREATE DATASET and DATALOADER')
    train_dataset = ImageDataset(config_file.DATA_FOLDER, file_index, file_ground_truth_train, config)
    val_dataset = ImageDataset(config_file.DATA_FOLDER, file_index, file_ground_truth_val, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    print('\nEXPERIMENT: ', str(experiment_id))
    model = models_baselines.Dieleman(config)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    for t in range(config['epochs']):
        print(f"Epoch {t+1}\n------------------")
        print(f"\tTraining\n")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        print(f"\tValidating\n")
        val_loss = test_loop(train_dataloader, model, loss_fn)
        scheduler.step(val_loss)
    print("done")

# def tf_define_model_and_cost(config):
#     # tensorflow: define the model
#     with tf.name_scope('model'):
#         x = tf.compat.v1.placeholder(tf.float32, [None, config['xInput'], config['yInput']])
#         y_ = tf.compat.v1.placeholder(tf.float32, [None, config['num_classes_dataset']])
#         is_train = tf.compat.v1.placeholder(tf.bool)
#         y = models.model_number(x, is_train, config)
#         normalized_y = tf.nn.sigmoid(y)
#         print(normalized_y.get_shape())
#     print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')
#
#     # tensorflow: define cost function
#     with tf.name_scope('metrics'):
#         # if you use softmax_cross_entropy be sure that the output of your model has linear units!
#         cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
#         if config['weight_decay'] != None:
#             vars = tf.trainable_variables()
#             lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name ])
#             cost = cost + config['weight_decay']*lossL2
#             print('L2 norm, weight decay!')
#
#     # print all trainable variables, for debugging
#     model_vars = [v for v in tf.global_variables()]
#     for variables in model_vars:
#         print(variables)
#
#     return [x, y_, is_train, y, normalized_y, cost]

# def data_gen(id, audio_repr_path, gt, pack):
#
#     [config, sampling, param_sampling, augmentation] = pack
#
#     # load audio representation -> audio_repr shape: NxM
#     audio_rep = pickle.load(open(config_file.DATA_FOLDER + audio_repr_path, 'rb'))
#     if config['pre_processing'] == 'logEPS':
#         audio_rep = np.log10(audio_rep + np.finfo(float).eps)
#     elif config['pre_processing'] == 'logC':
#         audio_rep = np.log10(10000 * audio_rep + 1)
#
#     # let's deliver some data!
#     last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
#     if sampling == 'random':
#         for i in range(0, param_sampling):
#             time_stamp = random.randint(0,last_frame-1)
#             yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)
#
#     elif sampling == 'overlap_sampling':
#         for time_stamp in range(0, last_frame, param_sampling):
#             yield dict(X = audio_rep[time_stamp : time_stamp+config['xInput'], : ], Y = gt, ID = id)
#
# if __name__ == '__main__':
#
#     # load config parameters defined in 'config_file.py'
#     parser = argparse.ArgumentParser()
#     parser.add_argument('configuration',
#                         help='ID in the config_file dictionary')
#     args = parser.parse_args()
#     config = config_file.config_train[args.configuration]
#
#     # load config parameters used in 'preprocess_librosa.py',
#     config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
#     with open(config_json, "r") as f:
#         params = json.load(f)
#     config['audio_rep'] = params
#
#     # set patch parameters
#     if config['audio_rep']['type'] == 'waveform':
#         raise ValueError('Waveform-based training is not implemented')
#
#     elif config['audio_rep']['spectrogram_type'] == 'mel':
#         config['xInput'] = config['n_frames']
#         config['yInput'] = config['audio_rep']['n_mels']
#
#     # load audio representation paths
#     file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
#     [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)
#
#     # load training data
#     file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
#     [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)
#
#     # load validation data
#     file_ground_truth_val = config_file.DATA_FOLDER + config['gt_val']
#     [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)
#
#     # set output
#     config['classes_vector'] = list(range(config['num_classes_dataset']))
#
#     print('# Train:', len(ids_train))
#     print('# Val:', len(ids_val))
#     print('# Classes:', config['classes_vector'])
#
#     # save experimental settings
#     experiment_id = str(shared.get_epoch_time()) + args.configuration
#     model_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#     json.dump(config, open(model_folder + 'config.json', 'w'))
#     print('\nConfig file saved: ' + str(config))
#
#     # tensorflow: define model and cost
#     [x, y_, is_train, y, normalized_y, cost] = tf_define_model_and_cost(config)
#
#     # tensorflow: define optimizer
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
#     with tf.control_dependencies(update_ops):
#         lr = tf.placeholder(tf.float32)
#         if config['optimizer'] == 'SGD_clip':
#             optimizer = tf.train.GradientDescentOptimizer(lr)
#             gradients, variables = zip(*optimizer.compute_gradients(cost))
#             gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#             train_step = optimizer.apply_gradients(zip(gradients, variables))
#         elif config['optimizer'] == 'SGD':
#             optimizer = tf.train.GradientDescentOptimizer(lr)
#             train_step = optimizer.minimize(cost)
#         elif config['optimizer'] == 'Adam':
#             optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#             train_step = optimizer.minimize(cost)
#
#     sess = tf.InteractiveSession()
#     tf.keras.backend.set_session(sess)
#
#     print('\nEXPERIMENT: ', str(experiment_id))
#     print('-----------------------------------')
#
#     # pescador train: define streamer
#     train_pack = [config, config['train_sampling'], config['param_train_sampling'], False]
#     train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train[id], train_pack) for id in ids_train]
#     train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
#     train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
#     train_batch_streamer = pescador.ZMQStreamer(train_batch_streamer)
#
#     # pescador val: define streamer
#     val_pack = [config, 'overlap_sampling', config['xInput'], False]
#     val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_val[id], val_pack) for id in ids_val]
#     val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
#     val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=config['val_batch_size'], partial=True)
#     val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)
#
#     # tensorflow: create a session to run the tensorflow graph
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     if config['load_model'] != None: # restore model weights from previously saved model
#         saver.restore(sess, config['load_model']) # end with /!
#         print('Pre-trained model loaded!')
#
#     # writing headers of the train_log.tsv
#     fy = open(model_folder + 'train_log.tsv', 'a')
#     fy.write('Epoch\ttrain_cost\tval_cost\tepoch_time\tlearing_rate\n')
#     fy.close()
#
#     # training
#     k_patience = 0
#     cost_best_model = np.Inf
#     tmp_learning_rate = config['learning_rate']
#     print('Training started..')
#     for i in range(config['epochs']):
#         # training: do not train first epoch, to see random weights behaviour
#         start_time = time.time()
#         array_train_cost = []
#         if i != 0:
#             for train_batch in train_batch_streamer:
#                 tf_start = time.time()
#                 _, train_cost = sess.run([train_step, cost],
#                                          feed_dict={x: train_batch['X'], y_: train_batch['Y'], lr: tmp_learning_rate, is_train: True})
#                 array_train_cost.append(train_cost)
#
#         # validation
#         array_val_cost = []
#         for val_batch in val_batch_streamer:
#             val_cost = sess.run([cost],
#                                 feed_dict={x: val_batch['X'], y_: val_batch['Y'], is_train: False})
#             array_val_cost.append(val_cost)
#
#         # Keep track of average loss of the epoch
#         train_cost = np.mean(array_train_cost)
#         val_cost = np.mean(array_val_cost)
#         epoch_time = time.time() - start_time
#         fy = open(model_folder + 'train_log.tsv', 'a')
#         fy.write('%d\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate))
#         fy.close()
#
#         # Decrease the learning rate after not improving in the validation set
#         if config['patience'] and k_patience >= config['patience']:
#             print('Changing learning rate!')
#             tmp_learning_rate = tmp_learning_rate / 2
#             print(tmp_learning_rate)
#             k_patience = 0
#
#         # Early stopping: keep the best model in validation set
#         if val_cost >= cost_best_model:
#             k_patience += 1
#             print('Epoch %d, train cost %g, val cost %g,'
#                   'epoch-time %gs, lr %g, time-stamp %s' %
#                   (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate,
#                    str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))
#
#         else:
#             # save model weights to disk
#             save_path = saver.save(sess, model_folder)
#             print('Epoch %d, train cost %g, val cost %g, '
#                   'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
#                   ' saved in: %s' %
#                   (i+1, train_cost, val_cost, epoch_time,tmp_learning_rate,
#                    str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
#             cost_best_model = val_cost
#
#     print('\nEVALUATE EXPERIMENT -> '+ str(experiment_id))
