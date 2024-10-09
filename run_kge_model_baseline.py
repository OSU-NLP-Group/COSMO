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
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except Exception as e:
    pass

# from model import DistMult, ConvE
from model_st import MKGE
from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as DatasetGeometric, DataLoader as DataLoaderGeometric

from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset_baseline import iWildCamOTTDataset
from pytorchtools import EarlyStopping
from gen_utils.train_utils import *

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def make_infinite(dataloader):
    while True:
        yield from dataloader

# @profile
def train(train_loader, model, optimizer, writer, args, epoch_id, scheduler):
    model.train()
    torch.set_grad_enabled(True)

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
    criterion_ce = nn.CrossEntropyLoss()

    for labeled_batch in tqdm(train_loader):
        # image, location, time, species_id = labeled_batch
        
        image, location, time, species_id = labeled_batch
        
        image = move_to(image, args.device)

        location = move_to(location, args.device)
        time = move_to(time, args.device)
        species_id = move_to(species_id, args.device)

        if args.dataset == 'mountain_zebra':
            location = None

        outputs = model.forward_ce(None, image, time, location)

        batch_results = {
            'y_true': species_id.cpu(),
            'y_pred': outputs.cpu(),
        }

        # compute objective
        loss = criterion_ce(batch_results['y_pred'], batch_results['y_true'])
        batch_results['objective'] = loss.item()
        loss.backward()

        avg_loss_image_id += loss.item()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)

        batch_idx += 1
        if args.debug:
            break

        # x = avg_loss_image_id/(batch_idx+1)
        # print(x)

    if scheduler is not None:
        scheduler.step()

    avg_loss_image_id = avg_loss_image_id/len(train_loader)
    print('train/avg_loss = {}'.format(avg_loss_image_id))
    writer.add_scalar('loss/train', avg_loss_image_id, epoch_id)

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

    
    results['epoch'] = epoch_id
    print(f'Train epoch {epoch_id}, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100.0:.2f}')
    
    writer.add_scalar('acc/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)
    writer.add_scalar('f1_macro/train', results[metrics[2].agg_metric_field]*100.0, epoch_id)

    return epoch_y_pred, epoch_y_true

def evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args):
    model.eval()
    torch.set_grad_enabled(False)
    criterion = nn.CrossEntropyLoss()

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
    for labeled_batch in tqdm(val_loader):
        image, location, time, species_id = labeled_batch
        
        image = move_to(image, args.device)

        location = move_to(location, args.device)

        time = move_to(time, args.device)
        species_id = move_to(species_id, args.device)

        if args.dataset == 'mountain_zebra':
            location = None

        outputs = model.forward_ce(None, image, time, location)

        batch_results = {
            'y_true': species_id.cpu(),
            'y_pred': outputs.cpu(),
        }

        batch_results['objective'] = criterion(batch_results['y_pred'], batch_results['y_true']).item()
        avg_loss_image_id += batch_results['objective']

        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)

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

    results['epoch'] = epoch_id

    avg_loss_image_id = avg_loss_image_id/len(val_loader)
    
    print('val/avg_loss = {}'.format(avg_loss_image_id))
    writer.add_scalar('loss/val', avg_loss_image_id, epoch_id)

    writer.add_scalar('acc/val', results[metrics[0].agg_metric_field]*100.0, epoch_id)
    writer.add_scalar('f1_macro/val', results[metrics[2].agg_metric_field]*100.0, epoch_id)

    print(f'Eval. epoch {epoch_id}, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100:.2f}')

    return epoch_y_pred, epoch_y_true


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

def calc_agg_results(epoch_y_pred_ilt, epoch_y_true_ilt, epoch_y_pred_it, epoch_y_true_it):
    epoch_y_pred_overall, epoch_y_true_overall = epoch_y_pred_ilt.tolist(), epoch_y_true_ilt.tolist()

    epoch_y_pred_overall.extend(epoch_y_pred_it.tolist())
    epoch_y_true_overall.extend(epoch_y_true_it.tolist())

    epoch_y_pred_overall, epoch_y_true_overall = torch.tensor(epoch_y_pred_overall), torch.tensor(epoch_y_true_overall)

    metrics = [
        Accuracy(prediction_fn=None),
        Recall(prediction_fn=None, average='macro'),
        F1(prediction_fn=None, average='macro'),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred_overall, epoch_y_true_overall),
                    })

    return results, metrics

'''
CUDA_VISIBLE_DEVICES=5 python run_kge_model_st.py --n_epochs 1
'''

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['iwildcam', 'mountain_zebra'], default='iwildcam')
    parser.add_argument('--data-dir', type=str, default='iwildcam_v2.0/')
    parser.add_argument('--img-dir', type=str, default='iwildcam_v2.0/imgs/')
    parser.add_argument('--iwildcam-image-h5-path', type=str, default='/local/scratch/pahuja.9/iwildcam2020_images.h5')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='default lr for all parameters')
    parser.add_argument('--loc-lr', type=float, default=1e-3, help='lr for location embedding')
    parser.add_argument('--time-lr', type=float, default=1e-3, help='lr for time embedding')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=813765)
    parser.add_argument('--save-dir', type=str, default='ckpts/toy/')
    parser.add_argument('--ckpt-path', type=str, default=None, help='path to ckpt for restarting expt')
    parser.add_argument('--start-epoch', type=int, default=0, help='epoch id to restore model')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='early stop if metric does not improve for x epochs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--use-loss-es', action='store_true', help='use val. loss for early stopping')
    parser.add_argument('--add-inverse-rels', action='store_true', help='add inverse relations for R-GCN/CompGCN')

    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')

    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--location_input_dim', type=int, default=2)
    parser.add_argument('--time_input_dim', type=int, default=2, help='2 corresponds to hour and month. change to 1 for just hour or month.')
    parser.add_argument('--location_time_input_dim', type=int, default=3)
    parser.add_argument('--mlp_location_numlayer', type=int, default=3)
    parser.add_argument('--mlp_time_numlayer', type=int, default=3)
    parser.add_argument('--mlp_location_time_numlayer', type=int, default=3)
    parser.add_argument('--loc-loss-coeff', type=float, default=1e0)
    parser.add_argument('--num-neg-frac', type=float, default=0.2)
    
    parser.add_argument('--use-distmult-model', action='store_true')
    parser.add_argument('--img-embed-model', choices=['resnet18', 'resnet50', 'inc-resnet-v2'], default='resnet50')
    parser.add_argument('--kg-embed-model', choices=['distmult', 'conve'], default='distmult')
    parser.add_argument('--use-subtree', action='store_true', help='use truncated OTT')
    parser.add_argument('--omit-taxon-ids', action='store_true', help='omit taxon ids in embedding')
    parser.add_argument('--use-h5', action='store_true', help='use hdf5 instead of raw images')
    parser.add_argument('--use-data-subset', action='store_true')
    parser.add_argument('--subset-size', type=int, default=10)
    parser.add_argument('--use-bce-for-location', action='store_true', help='use BCE loss for location')
    parser.add_argument('--use-ce-for-location', action='store_true', help='use CE loss for location')
    parser.add_argument('--use-bce-for-time', action='store_true', help='use BCE loss for time')
    parser.add_argument('--use-ce-for-time', action='store_true', help='use CE loss for time')
    parser.add_argument('--use-bce-for-location-time', action='store_true', help='use BCE loss for location-time')
    parser.add_argument('--use-ce-for-location-time', action='store_true', help='use CE loss for location-time')
    parser.add_argument('--use-cluster-centroids-for-location', action='store_true', help='use 6 cluster centroids for location')
    parser.add_argument('--use-learned-loc-embed', action='store_true', help='use learned embedding for location')

    parser.add_argument('--exclude-image-id', action='store_true', help='exclude image-id for training')
    parser.add_argument('--taxonomy-type', choices=['ott', 'standard'], default='ott')
    parser.add_argument('--add-reverse-id-id', action='store_true', help='add reversed triples for id-id')

    # options for img-time
    parser.add_argument('--only-hour', action='store_true', help='use only hour for img-time triples') 
    parser.add_argument('--only-month', action='store_true', help='use only month for img-time triples')
    parser.add_argument('--use-circular-space', action='store_true', help='use circular space for hour and month')

    # ConvE hyperparams
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--use-bias', action='store_true', default=True, help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

    # experimental
    parser.add_argument('--use-location-breakdown', action='store_true', help='break down location into hh,mm,ss for lat.,long.')
    parser.add_argument('--use-prop-sampling', action='store_true', help='mix all dataloader samples except (img, id) prop. acc. to dataset size')
    parser.add_argument('--use-uniform-sampling', action='store_true', help='mix all dataloader samples except (img, id) uniformly')
    parser.add_argument('--freeze-mlp', action='store_true')

    args = parser.parse_args()


    print('args = {}'.format(args))
    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

        
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    writer = SummaryWriter(log_dir=args.save_dir)

    # datacsv = pd.read_csv("../camera_trap/data_triples.csv")
    # datacsv = datacsv.loc[(datacsv["datatype_h"] == "image") & (datacsv["datatype_t"] == "id")]
    
    if args.dataset == 'iwildcam':
        if args.use_subtree:
            if args.taxonomy_type == 'standard':
                datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree_standard.csv'), low_memory=False)
            else:
                datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
        else:
            datacsv = pd.read_csv(os.path.join(args.data_dir, 'data_triples.csv'), low_memory=False)

        if args.use_subtree:
            if args.taxonomy_type == 'standard':
                entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree_standard_st.json')
            else:
                entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree_st.json')
        else:
            entity_id_file = os.path.join(args.data_dir, 'entity2id_st.json')
    else:
        datacsv = pd.read_csv(os.path.join(args.data_dir, 'data_triples.csv'), low_memory=False)
        entity_id_file = os.path.join(args.data_dir, 'entity2id.json')
    
    
    if not os.path.exists(entity_id_file):
        entity2id = {} # each of triple types have their own entity2id
        
        for i in tqdm(range(datacsv.shape[0])):
            if args.omit_taxon_ids and (datacsv.iloc[i,1] != 'image' or datacsv.iloc[i,4] != 'id'):
                continue

            if datacsv.iloc[i,1] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,0]))))

            if datacsv.iloc[i,-2] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,-3]))))
        json.dump(entity2id, open(entity_id_file, 'w'))
    else:
        entity2id = json.load(open(entity_id_file, 'r'))
    
    num_ent_id = len(entity2id)

    print('len(entity2id) = {}'.format(len(entity2id)))

    target_list = generate_target_list(datacsv, entity2id)

    train_ILT_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, disjoint=False, is_train=True)
    print('len(train_ILT_dataset) = {}'.format(len(train_ILT_dataset)))


    val_ILT_dataset = iWildCamOTTDataset(datacsv, 'val', args, entity2id, target_list, disjoint=True)
    print('len(val_ILT_dataset) = {}'.format(len(val_ILT_dataset)))


    model_kwargs = {}
    if args.kg_embed_model == 'conve':
        model_kwargs['drop_last'] = True
    
    train_loader = DataLoader(
        train_ILT_dataset,
        shuffle=True, # Shuffle training dataset
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        **model_kwargs)
    
    val_loader = DataLoader(
        val_ILT_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    
    kwargs = {}

    model = MKGE(args, num_ent_id, target_list, args.device, **kwargs)

    model.to(args.device)
    
    if args.freeze_mlp:
        for param in model.mlp.parameters():
            param.requires_grad = False

    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True, ckpt_path=os.path.join(args.save_dir, 'model.pt'), best_ckpt_path=os.path.join(args.save_dir, 'best_model.pt'))

    params_diff_lr = ['ent_embedding', 'image_embedding', 'location_embedding', 'time_embedding']

    optimizer_grouped_parameters = [
            {"params": [param for p_name, param in model.named_parameters() if not any([x in p_name for x in params_diff_lr])]},
            {"params": model.ent_embedding.parameters(), "lr": args.lr},
            {"params": model.image_embedding.parameters(), "lr": 3e-5},
            {"params": model.location_embedding.parameters(), "lr": args.loc_lr},
            {"params": model.time_embedding.parameters(), "lr": args.time_lr},
        ]

    n_params_model = sum(torch.numel(param) for p_name, param in model.named_parameters())
    n_params_optimizer = sum([sum([torch.numel(x) for x in group['params']]) for group in optimizer_grouped_parameters])

    print('n_params_model = {}'.format(n_params_model))
    print('n_params_optimizer = {}'.format(n_params_optimizer))

    assert n_params_model == n_params_optimizer

    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay)
        scheduler = None

    elif args.optimizer == 'adamw':
        # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs*5, eta_min=1e-6)


    else:
        raise NotImplementedError

    # restore from ckpt
    if args.ckpt_path:
        print('ckpt loaded...')
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['dense_optimizer'])

    for epoch_id in range(args.start_epoch, args.n_epochs):
        print('\nEpoch [%d]:\n' % epoch_id)

        # First run training
        epoch_y_pred, epoch_y_true = train(train_loader, model, optimizer, writer, args, epoch_id, scheduler)

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

        print(f'Train epoch {epoch_id}, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100.0:.2f}')
    
        writer.add_scalar('acc/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)
        writer.add_scalar('f1_macro/train', results[metrics[2].agg_metric_field]*100.0, epoch_id)

        # Then run val
        epoch_y_pred, epoch_y_true = evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args)

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(epoch_y_pred, epoch_y_true),
                        })

        if args.use_loss_es:
            early_stopping(-1*results[metrics[2].agg_metric_field], model, optimizer)
        else:
            early_stopping(-1*results[metrics[0].agg_metric_field], model, optimizer)

        writer.add_scalar('acc/val', results[metrics[0].agg_metric_field]*100.0, epoch_id)
        writer.add_scalar('f1_macro/val', results[metrics[2].agg_metric_field]*100.0, epoch_id)

        print(f'Eval. epoch {epoch_id}, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100:.2f}')

        if early_stopping.early_stop:
            print("Early stopping...")
            break

    writer.close()

if __name__=='__main__':
    main()
