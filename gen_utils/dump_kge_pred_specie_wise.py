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

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset import iWildCamOTTDataset

def evaluate(model, val_loader, target_list, node_parent_map, args):
    model.eval()
    torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0

    y_pred_dict = {}

    for label_id in range(182):
        y_pred_dict[label_id] = []

    for labeled_batch in tqdm(val_loader):
        h, r, t = labeled_batch
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, t, triple_type=('image', 'id'))

        batch_results = {
            'y_true': t.cpu(),
            'y_pred': outputs.cpu(),
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
        Recall(prediction_fn=None, average='macro'),
        F1(prediction_fn=None, average='macro'),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    print(f'Eval., split: {args.split}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100:.2f}')

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
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--location_input_dim', type=int, default=2)
    parser.add_argument('--time_input_dim', type=int, default=1)
    parser.add_argument('--mlp_location_numlayer', type=int, default=3)
    parser.add_argument('--mlp_time_numlayer', type=int, default=3)

    parser.add_argument('--img-embed-model', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--use-data-subset', action='store_true')
    parser.add_argument('--subset-size', type=int, default=10)

    parser.add_argument('--kg-embed-model', choices=['distmult', 'conve'], default='distmult')
    
    # ConvE hyperparams
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--use-bias', action='store_true', default=True, help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

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

    entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree.json')

    if not os.path.exists(entity_id_file):
        entity2id = {} # each of triple types have their own entity2id
        
        for i in tqdm(range(datacsv.shape[0])):
            if datacsv.iloc[i,1] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,0]))))

            if datacsv.iloc[i,-2] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,-3]))))
        json.dump(entity2id, open(entity_id_file, 'w'))
    else:
        entity2id = json.load(open(entity_id_file, 'r'))
    
    num_ent_id = len(entity2id)

    print('len(entity2id) = {}'.format(len(entity2id)))
    
    # print('entity2id = {}'.format(entity2id))
    id2entity = {v:k for k,v in entity2id.items()}

    target_list = generate_target_list(datacsv, entity2id)

    val_image_to_id_dataset = iWildCamOTTDataset(datacsv, args.split, args, entity2id, target_list, head_type="image", tail_type="id")
    print('len(val_image_to_id_dataset) = {}'.format(len(val_image_to_id_dataset)))

    val_loader = DataLoader(
        val_image_to_id_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True)

    model = MKGE(args, num_ent_id, target_list, args.device)

    model.to(args.device)

    # restore from ckpt
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    y_pred_dict = evaluate(model, val_loader, target_list, node_parent_map, args)

    json.dump(y_pred_dict, open(os.path.join(args.out_dir, 'y_pred_dict_{}.json'.format(args.split)), 'w'))


