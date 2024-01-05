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
from collections import defaultdict, Counter
import math

sys.path.append('../')

from model import MKGE
from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset import iWildCamOTTDataset

def level(a, node_parent_map):
    if a not in node_parent_map:
        return 0
    parent = node_parent_map[a]
    return level(parent, node_parent_map)+1

def height(a, parent_node_map):
    ans = -1

    if a not in parent_node_map:
        return 0

    for child in parent_node_map[a]:
        ans = max(ans, height(child, parent_node_map))

    return ans+1


def least_common_ancestor(a, b, node_parent_map):

    if level(a, node_parent_map) > level(b, node_parent_map):
        a, b = b, a

    # if both are not at same level then move lower node upwards
    d = level(b, node_parent_map) - level(a, node_parent_map)
    
    # node_parent_map[i] stores the parent of node i 
    while d > 0:
        b = node_parent_map[b]
        d-= 1
    
    # base case if one was the ancestor of other node
    if a == b:
        return a
    
    # print('a = {}, b = {}'.format(a, b))
    if a not in node_parent_map or b not in node_parent_map:
        return '805080' # return root as ancestor

    while node_parent_map[a] != node_parent_map[b]:
        a = node_parent_map[a]
        b = node_parent_map[b]

        # print('flag 1')
 
    return node_parent_map[a]


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

def evaluate(model, val_loader, id2entity, overall_id_to_name, taxon_id_to_name, target_list, node_parent_map, parent_node_map, args):
    model.eval()
    torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    correct_idx = []

    avg_lca_height = 0
    total = 0

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

        b_range = torch.arange(y_pred.size()[0], device=args.device)

        arg_outputs = torch.argsort(outputs, dim=-1, descending=True)
        rank = 1 + torch.argsort(arg_outputs, dim=-1, descending=False)[b_range, y_true]
        # print('rank = {}'.format(rank))

        for i in range(y_true.size(0)):
            if y_pred[i] == y_true[i]:
                correct_idx.append(batch_idx * args.batch_size + i)
            else:
                lca = least_common_ancestor(int(id2entity[target_list[y_pred[i]].item()]), int(id2entity[target_list[y_true[i]].item()]), node_parent_map)
                lca_height = height(lca, parent_node_map)

                avg_lca_height += lca_height
                total += 1

        epoch_y_pred.append(y_pred)

        batch_idx += 1
        if args.debug and batch_idx>10:
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
    
    avg_lca_height = avg_lca_height/total

    # print('total = {}'.format(total))
    # print('avg_lca_height = {}'.format(avg_lca_height))
    
    return correct_idx, epoch_y_pred.tolist(), epoch_y_true.tolist(), avg_lca_height

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

def check_list_equal(list_1, list_2):
    for i in range(len(list_1)):
        if list_1[i] != list_2[i]:
            return False
    return True


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../iwildcam_v2.0/')
    parser.add_argument('--img-dir', type=str, default='../iwildcam_v2.0/imgs/')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--seed', type=int, default=813765)

    parser.add_argument('--ckpt-1-path', type=str, default=None, help='path to ckpt 1 for restarting expt')
    parser.add_argument('--ckpt-2-path', type=str, default=None, help='path to ckpt 1 for restarting expt')

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
    parser.add_argument('--add-id-id', action='store_true', help='add idtoid triples in addition to other triples for training')

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
    parent_node_map = defaultdict(list)

    for idx in range(len(datacsv_id_id)):
        node = int(float(datacsv.iloc[idx, 0]))
        parent = int(float(datacsv.iloc[idx, -3]))

        node_parent_map[node] = parent
        parent_node_map[parent].append(node)

    # print('node_parent_map = {}'.format(node_parent_map))
    # sys.exit(0)

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
    # print('target_list = {}'.format(target_list))

    val_image_to_id_dataset = iWildCamOTTDataset(datacsv, args.split, args, entity2id, target_list, head_type="image", tail_type="id")
    print('len(val_image_to_id_dataset) = {}'.format(len(val_image_to_id_dataset)))

    val_loader = DataLoader(
        val_image_to_id_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True)

 
    model_1 = MKGE(args, num_ent_id, target_list, args.device)
    model_2 = MKGE(args, num_ent_id, target_list, args.device)

    model_1.to(args.device)
    model_2.to(args.device)

    overall_id_to_name = json.load(open(os.path.join(args.data_dir, 'overall_id_to_name.json'), 'r'))
    taxon_id_to_name = json.load(open(os.path.join(args.data_dir, 'taxon_id_to_name.json'), 'r'))

    taxon_id_to_name['8032203'] = 'empty'

    # restore from ckpt
    if args.ckpt_1_path:
        ckpt = torch.load(args.ckpt_1_path, map_location=args.device)
        model_1.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    if args.ckpt_2_path:
        ckpt = torch.load(args.ckpt_2_path, map_location=args.device)
        model_2.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    model_1_correct_idx, model_1_pred, model_1_true, lca_model_1 = evaluate(model_1, val_loader, id2entity, overall_id_to_name, taxon_id_to_name, target_list, node_parent_map, parent_node_map, args)

    model_2_correct_idx, model_2_pred, model_2_true, lca_model_2 = evaluate(model_2, val_loader, id2entity, overall_id_to_name, taxon_id_to_name, target_list, node_parent_map, parent_node_map, args)

    print('lca_model_1 = {}'.format(lca_model_1))
    print('lca_model_2 = {}'.format(lca_model_2))

    # model_1 - model_2
    model_1_correct = list(set(model_1_correct_idx) - set(model_2_correct_idx))

    # print('len(model_1_correct) = {}'.format(len(model_1_correct)))

    assert check_list_equal(model_1_true, model_2_true)

    # show taxonomy for cases where model_1 is correct but model_2 is not

    true_pred_c = Counter()

    for idx in model_1_correct:
        model_1_true_label = model_1_true[idx]
        model_1_pred_label = model_1_pred[idx]
        model_2_pred_label = model_2_pred[idx]

        assert model_1_true_label == model_1_pred_label # model_1 is correct for this example, model_2 is incorrect

        print('true_label = {}, model_1_pred_label = {}, model_2_pred_label = {}'.format(overall_id_to_name[id2entity[target_list[model_1_true_label].item()]], overall_id_to_name[id2entity[target_list[model_1_pred_label].item()]], overall_id_to_name[id2entity[target_list[model_2_pred_label].item()]]))

        true_pred_c.update([(overall_id_to_name[id2entity[target_list[model_1_true_label].item()]], overall_id_to_name[id2entity[target_list[model_2_pred_label].item()]])])
        
        # print taxonomy (list of ancestors) for y_true
        print('ancestors of y_true: ')
        print_ancestors(int(id2entity[target_list[model_1_true_label].item()]), node_parent_map, target_list, taxon_id_to_name, overall_id_to_name)

        print('ancestors of y_pred: ')
        print_ancestors(int(id2entity[target_list[model_2_pred_label].item()]), node_parent_map, target_list, taxon_id_to_name, overall_id_to_name)

        print('\n')

    print(true_pred_c.most_common())
    


    

