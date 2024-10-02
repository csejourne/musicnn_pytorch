import numpy as np
import pandas as pd
import torch
import pickle as pk
import os
from datetime import datetime
from sklearn import metrics
from torch.utils.data import Dataset
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')



class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

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
        img_file.close()
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

