import argparse
import json
import config_file, shared, train
import numpy as np
import torch
import models
from torch.utils.data import DataLoader
from torch import nn


TEST_BATCH_SIZE = 128
FILE_INDEX = config_file.DATA_FOLDER + 'audio_representation/'+config_file.DATASET+'__time-freq/index.tsv'
FILE_GROUND_TRUTH_TEST = config_file.DATA_FOLDER + 'index/'+config_file.DATASET+'/test_gt_'+config_file.DATASET+'.tsv'

def evaluation(dataloader, model, loss_fn):
    """
    returns:
        - `preds`: tensor
        - `ground_truth`: tensor
        - `test_loss`: float
    """
    # Important for BN
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    preds = []
    ground_truth = []
    with torch.no_grad():
        for X, y in dataloader:
            preds_batch = model(X)
            test_loss += loss_fn(preds_batch, y).item()
            preds.append(preds_batch)
            ground_truth.append(y)
    test_loss /= num_batches
    preds = torch.concatenate(preds)
    preds = nn.Sigmoid()(preds)
    ground_truth = torch.concatenate(ground_truth)
    return preds, ground_truth, test_loss

if __name__ == '__main__':

    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list', nargs='+', help='List of models to evaluate', required=True)
    args = parser.parse_args()
    models_list = args.list

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(FILE_INDEX)

    # load ground truth
    [ids, id2gt] = shared.load_id2gt(FILE_GROUND_TRUTH_TEST)
    print('# Test set', len(ids))

    for i, model in enumerate(models_list):

        experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(model) + '/'
        config = json.load(open(experiment_folder + 'config.json'))
        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # Create test `Dataset` and `Dataloader`
        test_dataset = shared.ImageDataset(config_file.DATA_FOLDER, FILE_INDEX, FILE_GROUND_TRUTH_TEST, config)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

        # Restore model
        model_torch = models.select_model(config)
        model_torch.load_state_dict(torch.load(experiment_folder + 'model_weights.pth'))

        # Evaluate model
        print('Computing predictions\n')
        preds, ground_truth, test_loss = evaluation(test_dataloader,
                                                    model_torch,
                                                    nn.BCEWithLogitsLoss()
                                                    )

    print('Predictions computed, now evaluating..')
    roc_auc, pr_auc = shared.compute_auc(ground_truth, preds)

    # print experimental results
    print('\nExperiment: ' + str(models_list))
    print(config)
    print('\nROC-AUC: ' + str(roc_auc))
    print('\nPR-AUC: ' + str(pr_auc))
    # store experimental results
    to = open(experiment_folder + 'experiment.result', 'w')
    to.write('Experiment: ' + str(models_list))
    to.write('\nAUC: ' + str(roc_auc))
    to.write('\nAUC: ' + str(pr_auc))
    to.close()

# def evaluation(batch_dispatcher, tf_vars, array_cost, pred_array, id_array):
#
#     [sess, normalized_y, cost, x, y_, is_train] = tf_vars
#     for batch in tqdm(batch_dispatcher):
#         pred, cost_pred = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y'], is_train: False})
#         if not array_cost: # if array_cost is empty, is the first iteration
#             pred_array = pred
#             id_array = batch['ID'] 
#         else:
#             pred_array = np.concatenate((pred_array,pred), axis=0)
#             id_array = np.append(id_array,batch['ID'])
#         array_cost.append(cost_pred) 
#     print('predictions', pred_array.shape)          
#     print('cost', np.mean(array_cost))   
#     return array_cost, pred_array, id_array
#
# if __name__ == '__main__':
#
#     # which experiment we want to evaluate?
#     # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-l','--list', nargs='+', help='List of models to evaluate', required=True)
#     args = parser.parse_args()
#     models = args.list
#
#     # load all audio representation paths
#     [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(FILE_INDEX)
#
#     # load ground truth
#     [ids, id2gt] = shared.load_id2gt(FILE_GROUND_TRUTH_TEST)
#     print('# Test set', len(ids))
#
#     graphs = []
#     for i in range(len(models)):
#         graphs.append(tf.Graph())
#
#     array_cost, pred_array, id_array = [], None, None
#
#     for i, model in enumerate(models):
#
#         experiment_folder = config_file.DATA_FOLDER + 'experiments/' + str(model) + '/'
#         config = json.load(open(experiment_folder + 'config.json'))
#         print('Experiment: ' + str(model))
#         print('\n' + str(config))
#
#         # pescador: define (finite, batched & parallel) streamer
#         pack = [config, 'overlap_sampling', config['n_frames'], False]
#         streams = [pescador.Streamer(train.data_gen, id, id2audio_repr_path[id], id2gt[id], pack) for id in ids]
#         mux_stream = pescador.ChainMux(streams, mode='exhaustive')
#         batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True)
#         batch_streamer = pescador.ZMQStreamer(batch_streamer)    
#
#         # tensorflow: define model and cost
#         with graphs[i].as_default():
#             sess = tf.Session()
#             [x, y_, is_train, y, normalized_y, cost] = train.tf_define_model_and_cost(config)
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver()
#             results_folder = experiment_folder
#             saver.restore(sess, results_folder)
#             tf_vars = [sess, normalized_y, cost, x, y_, is_train]
#             array_cost, pred_array, id_array = evaluation(batch_streamer, tf_vars, array_cost, pred_array, id_array)
#             sess.close()
#
#     print('Predictions computed, now evaluating..')
#     roc_auc, pr_auc = shared.auc_with_aggergated_predictions(pred_array, id_array, ids, id2gt)
#
#     # print experimental results
#     print('\nExperiment: ' + str(models))
#     print(config)
#     print('ROC-AUC: ' + str(roc_auc))
#     print('PR-AUC: ' + str(pr_auc))
#     # store experimental results
#     to = open(experiment_folder + 'experiment.result', 'w')
#     to.write('Experiment: ' + str(models))
#     to.write('\nAUC: ' + str(roc_auc))
#     to.write('\nAUC: ' + str(pr_auc))
#     to.close()
