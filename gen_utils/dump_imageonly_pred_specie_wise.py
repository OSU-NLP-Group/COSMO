import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
import json
from collections import defaultdict
import math

sys.path.append('../')

from model import MKGE
from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy
from PIL import Image
from dataset import iWildCamOTTDataset
import torchvision.transforms as transforms

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def print_ancestors(species_id, node_parent_map, target_list, taxon_id_to_name, overall_id_to_name):
    out = []
    curr_node = species_id

    while True:
        if str(curr_node) in taxon_id_to_name:
            out.append(taxon_id_to_name[str(curr_node)])
        else:
            out.append(overall_id_to_name[str(curr_node)])
            break

        if curr_node not in node_parent_map:
            break
        curr_node = node_parent_map[curr_node]

    print(' --> '.join(out))

def evaluate(model, val_loader, args):
    model.eval()
    torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0

    y_pred_dict = {}

    for label_id in range(182):
        y_pred_dict[label_id] = []

    for labeled_batch in tqdm(val_loader):
        x, y_true = labeled_batch
        x = move_to(x, args.device)
        y_true = move_to(y_true, args.device)

        outputs = model(x)

        batch_results = {
            # 'g': g,
            'y_true': y_true.cpu(),
            'y_pred': outputs.cpu(),
            # 'metadata': metadata,
        }

        y_true = detach_and_clone(batch_results['y_true'])
        epoch_y_true.append(y_true)
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)


        epoch_y_pred.append(y_pred)

        for i in range(y_true.size(0)):
            x = (y_pred[i] == y_true[i]).long().item()
            y_pred_dict[y_true[i].item()].append(x) # 1 means prediction matches label, 0 otherwise. Used for calculating F1 score.

        batch_idx += 1
        if args.debug:
            break

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)

    metrics = [
        Accuracy(prediction_fn=None),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    print(f'Eval., split: {args.split}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}')

    return y_pred_dict

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def generate_target_list(data, entity2id):
    sub = data.loc[(data["datatype_h"] == "image") & (data["datatype_t"] == "id"), ['t']]
    sub = list(sub['t'])
    categories = []
    for item in tqdm(sub):
        if entity2id[str(int(float(item)))] not in categories:
            categories.append(entity2id[str(int(float(item)))])
    # print('categories = {}'.format(categories))
    print("No. of target categories = {}".format(len(categories)))
    return torch.tensor(categories, dtype=torch.long).unsqueeze(-1)

class iWildCamDataset(Dataset):
    def __init__(self, datacsv, root, img_dir, mode, entity2id, target_list):  # dic_data <- datas
        super(iWildCamDataset, self).__init__()
        self.mode = mode
        self.datacsv = datacsv.loc[datacsv['split'] == mode, :]
        self.root = root
        self.img_dir = img_dir
        self.entity2id = entity2id
        self.target_list = target_list
        self.entity_to_species_id = {self.target_list[i, 0].item():i for i in range(len(self.target_list))}

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        y = torch.tensor([self.entity_to_species_id[self.entity2id[str(int(float(self.datacsv.iloc[idx, -3])))]]], dtype=torch.long).squeeze()

        img = Image.open(os.path.join(self.img_dir, self.datacsv.iloc[idx, 0])).convert('RGB')

        transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
        x = transform_steps(img)

        return x, y

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../iwildcam_v2.0/')
    parser.add_argument('--img-dir', type=str, default='../iwildcam_v2.0/imgs/')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--seed', type=int, default=813765)
    parser.add_argument('--ckpt-path', type=str, default=None, help='path to ckpt for restarting expt')
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--use-subtree', action='store_true', help='use truncated OTT')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--location_input_dim', type=int, default=2)
    parser.add_argument('--time_input_dim', type=int, default=1)
    parser.add_argument('--mlp_location_numlayer', type=int, default=3)
    parser.add_argument('--mlp_time_numlayer', type=int, default=3)

    parser.add_argument('--img-embed-model', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--use-data-subset', action='store_true')
    parser.add_argument('--subset-size', type=int, default=10)

    args = parser.parse_args()

    print('args = {}'.format(args))
    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)

    # construct OTT parent map
    datacsv_id_id = datacsv.loc[(datacsv['datatype_h'] == 'id') & (datacsv['datatype_t'] == 'id')]
    node_parent_map = {}

    for idx in range(len(datacsv_id_id)):
        node = int(float(datacsv.iloc[idx, 0]))
        parent = int(float(datacsv.iloc[idx, -3]))

        node_parent_map[node] = parent

    datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'))
    datacsv = datacsv.loc[(datacsv["datatype_h"] == "image") & (datacsv["datatype_t"] == "id")]

    entity2id = {} # each of triple types have their own entity2id
        
    for i in tqdm(range(datacsv.shape[0])):
        _get_id(entity2id, str(int(float(datacsv.iloc[i,-3]))))

    print('len(entity2id) = {}'.format(len(entity2id)))

    target_list = generate_target_list(datacsv, entity2id)

    val_dataset = iWildCamDataset(datacsv, os.path.join('iwildcam_v2.0', 'imgs/'), args.img_dir, args.split, entity2id, target_list)
    
    id2entity = {v:k for k,v in entity2id.items()}

    val_loader = DataLoader(
        val_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True)

    model = Resnet50(args)
    model.to(args.device)

    # restore from ckpt
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    y_pred_dict = evaluate(model, val_loader, args)

    json.dump(y_pred_dict, open(os.path.join(args.out_dir, 'y_pred_dict_{}.json'.format(args.split)), 'w'))


