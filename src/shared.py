import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    def __init__(self, data_folder, file_index, file_ground_truth, config, transform=None, target_transform=None):
        self.config = config
        self.data_folder = data_folder
        self.ids, self.id2gt = load_id2gt(file_ground_truth) # `ids` are `str`
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


def get_epoch_time():
    return int((datetime.now() - datetime(1970,1,1)).total_seconds())


def count_params(trainable_variables):
    # to return number of trainable variables. Example: shared.count_params(tf.trainable_variables()))
    return np.sum([np.prod(v.get_shape().as_list()) for v in trainable_variables])


def load_id2gt(gt_file):
    ids = []
    fgt = open(gt_file)
    id2gt = dict()
    for line in fgt.readlines():
        id, gt = line.strip().split("\t") # id is string
        id2gt[id] = eval(gt) # gt is array
        ids.append(id)
    return ids, id2gt


def load_id2path(index_file):
    paths = []
    fspec = open(index_file)
    id2path = dict()
    for line in fspec.readlines():
        id, path, _ = line.strip().split("\t")
        id2path[id] = path
        paths.append(path)
    return paths, id2path


def auc_with_aggergated_predictions(pred_array, id_array, ids, id2gt): 
    # averaging probabilities -> one could also do majority voting
    y_pred = []
    y_true = []
    for id in ids:
        try:
            avg = np.mean(pred_array[np.where(id_array==id)], axis=0)
            y_pred.append(avg)
            y_true.append(id2gt[id])
        except:
            print(id)
            
    print('Predictions are averaged, now computing AUC..')
    roc_auc, pr_auc = compute_auc(y_true, y_pred)
    return  np.mean(roc_auc), np.mean(pr_auc)
   

def compute_auc(true,estimated):
    pr_auc=[]
    roc_auc=[]    
    estimated = np.array(estimated)
    true = np.array(true) 
    for count in range(0,estimated.shape[1]):
        pr_auc.append(metrics.average_precision_score(true[:,count],estimated[:,count]))
        roc_auc.append(metrics.roc_auc_score(true[:,count],estimated[:,count]))
    return roc_auc, pr_auc

