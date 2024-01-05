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


from model import MKGE
from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy
from PIL import Image
from dataset import iWildCamOTTDataset
from pytorchtools import EarlyStopping

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

#################
# image to id
#################
    
def train_image_id(train_loader, model, optimizer, writer, args, epoch_id):
    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
    criterion_ce = nn.CrossEntropyLoss()

    for labeled_batch in tqdm(train_loader['image_to_id']):
        h, r, t = labeled_batch
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, t, triple_type=('image', 'id'))
        # outputs = model(h)

        batch_results = {
            'y_true': t.cpu(),
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

    avg_loss_image_id = avg_loss_image_id/len(train_loader['image_to_id'])
    print('train/avg_loss_image_id = {}'.format(avg_loss_image_id))
    writer.add_scalar('image_id_loss/train', avg_loss_image_id, epoch_id)

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

    
    results['epoch'] = epoch_id
    print(f'Train epoch {epoch_id}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}')
    
    writer.add_scalar('acc_image_id/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)

#################
# id to id
#################
def train_id_id(train_loader, model, optimizer, writer, args, epoch_id):
    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_id_id = 0.0
    criterion_ce = nn.CrossEntropyLoss()

    for labeled_batch in tqdm(train_loader['id_to_id']):
        h, r, t = labeled_batch
        # print(h, r, t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, t, triple_type=('id', 'id'))

        batch_results = {
            'y_true': t.cpu(),
            'y_pred': outputs.cpu(),
        }

        # compute objective
        loss = criterion_ce(batch_results['y_pred'], batch_results['y_true'])
        avg_loss_id_id += loss.item()

        # print('loss = {}'.format(loss.item()))
        batch_results['objective'] = loss.item()
        loss.backward()

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

    avg_loss_id_id = avg_loss_id_id/len(train_loader['id_to_id'])
    print('avg_loss_id_id = {}'.format(avg_loss_id_id))
    writer.add_scalar('avg_loss_id_id/train', avg_loss_id_id, epoch_id)

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

    results['epoch'] = epoch_id
    print(f'Train epoch {epoch_id}, id to id, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}')
    writer.add_scalar('acc_id_id/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)

#################
# image to location
#################
def train_image_location(train_loader, model, optimizer, writer, args, epoch_id):
    batch_idx = 0
    avg_loss_image_location = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for labeled_batch in tqdm(train_loader['image_to_location']):
        h, r, t = labeled_batch

        # print(h, r, t)
        # print(t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)
        
        outputs = model.forward_ce(h, r, t, triple_type=('image', 'location'))
        target = F.one_hot(t, num_classes=len(model.all_locs)).float()
        loss = criterion_bce(outputs, target)
        
        avg_loss_image_location += loss.item()

        loss.backward()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        batch_idx += 1
        if args.debug:
            break

    avg_loss_image_location = avg_loss_image_location/len(train_loader['image_to_location'])
    print('avg_loss_image_location = {}'.format(avg_loss_image_location))
    writer.add_scalar('avg_loss_image_location/train', avg_loss_image_location, epoch_id)

#################
# image to time
#################
def train_image_time(train_loader, model, optimizer, writer, args, epoch_id):
    batch_idx = 0
    avg_loss_image_time = 0.0
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    
    for labeled_batch in tqdm(train_loader['image_to_time']):
        h, r, t = labeled_batch

        # print(h, r, t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)
        
        outputs = model.forward_ce(h, r, t, triple_type=('image', 'time'))
        target = F.one_hot(t, num_classes=len(model.all_timestamps)).float()
        loss = criterion_bce(outputs, target)
    
        avg_loss_image_time += loss.item()

        loss.backward()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        batch_idx += 1
        if args.debug:
            break

    avg_loss_image_time = avg_loss_image_time/len(train_loader['image_to_time'])
    print('avg_loss_image_time = {}'.format(avg_loss_image_time))
    writer.add_scalar('avg_loss_image_time/train', avg_loss_image_time, epoch_id)

def train(model, train_loader, optimizer, epoch_id, writer, args):
    model.train()
    torch.set_grad_enabled(True)
    
    if args.add_id_id:
        train_id_id(train_loader, model, optimizer, writer, args, epoch_id)
    
    if args.add_image_location:
        train_image_location(train_loader, model, optimizer, writer, args, epoch_id)

    if args.add_image_time:
        train_image_time(train_loader, model, optimizer, writer, args, epoch_id)
    
    train_image_id(train_loader, model, optimizer, writer, args, epoch_id)
   
    return

def evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args):
    model.eval()
    torch.set_grad_enabled(False)
    criterion = nn.CrossEntropyLoss()

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
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
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    results['epoch'] = epoch_id

    avg_loss_image_id = avg_loss_image_id/len(val_loader)
    
    early_stopping(-1*results[metrics[0].agg_metric_field], model, optimizer)
    
    print('val/avg_loss_image_id = {}'.format(avg_loss_image_id))
    writer.add_scalar('image_id_loss/val', avg_loss_image_id, epoch_id)

    writer.add_scalar('acc_image_id/val', results[metrics[0].agg_metric_field]*100, epoch_id)

    print(f'Eval. epoch {epoch_id}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}')

    return results, epoch_y_pred


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

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['iwildcam', 'mountain_zebra'], default='iwildcam')
    parser.add_argument('--data-dir', type=str, default='iwildcam_v2.0/')
    parser.add_argument('--img-dir', type=str, default='iwildcam_v2.0/imgs/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=12)
    parser.add_argument('--img-lr', type=float, default=3e-5, help='lr for img embed params')
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

    parser.add_argument('--kg-embed-model', choices=['distmult', 'conve'], default='distmult')
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--location_input_dim', type=int, default=2)
    parser.add_argument('--time_input_dim', type=int, default=1)
    parser.add_argument('--mlp_location_numlayer', type=int, default=3)
    parser.add_argument('--mlp_time_numlayer', type=int, default=3)
    
    parser.add_argument('--img-embed-model', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--use-data-subset', action='store_true')
    parser.add_argument('--subset-size', type=int, default=10)

    parser.add_argument('--add-id-id', action='store_true', help='add idtoid triples in addition to other triples for training')
    parser.add_argument('--add-image-location', action='store_true', help='add imagetolocation triples in addition to other triples for training')
    parser.add_argument('--add-image-time', action='store_true', help='use only imagetotime triples in addition to other triples for training')

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

    writer = SummaryWriter(log_dir=args.save_dir)

    if args.dataset == 'iwildcam':
        datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
        entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree.json')
    else:
        datacsv = pd.read_csv(os.path.join(args.data_dir, 'data_triples.csv'), low_memory=False)
        entity_id_file = os.path.join(args.data_dir, 'entity2id.json')


    if not os.path.exists(entity_id_file):
        entity2id = {} # each of triple types have their own entity2id
        
        for i in tqdm(range(datacsv.shape[0])):
            if datacsv.iloc[i,1] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,0]))))

            if datacsv.iloc[i,4] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,3]))))
        json.dump(entity2id, open(entity_id_file, 'w'))
    else:
        entity2id = json.load(open(entity_id_file, 'r'))
    
    num_ent_id = len(entity2id)

    print('len(entity2id) = {}'.format(len(entity2id)))

    target_list = generate_target_list(datacsv, entity2id)

    train_image_to_id_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="id")
    print('len(train_image_to_id_dataset) = {}'.format(len(train_image_to_id_dataset)))

    if args.add_id_id:
        train_id_to_id_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="id", tail_type="id")
        print('len(train_id_to_id_dataset) = {}'.format(len(train_id_to_id_dataset)))

    if args.add_image_location:
        train_image_to_location_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="location")
        print('len(train_image_to_location_dataset) = {}'.format(len(train_image_to_location_dataset)))

    if args.add_image_time:
        train_image_to_time_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="time")
        print('len(train_image_to_time_dataset) = {}'.format(len(train_image_to_time_dataset)))

    val_image_to_id_dataset = iWildCamOTTDataset(datacsv, 'val', args, entity2id, target_list, head_type="image", tail_type="id")
    print('len(val_image_to_id_dataset) = {}'.format(len(val_image_to_id_dataset)))

    model_kwargs = {}
    if args.kg_embed_model == 'conve':
        model_kwargs['drop_last'] = True

    train_loader_image_to_id = DataLoader(
        train_image_to_id_dataset,
        shuffle=True, # Shuffle training dataset
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        **model_kwargs)

    if args.add_id_id:
        train_loader_id_to_id = DataLoader(
            train_id_to_id_dataset,
            shuffle=True, # Shuffle training dataset
            sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            **model_kwargs)

    if args.add_image_location:
        train_loader_image_to_location = DataLoader(
            train_image_to_location_dataset,
            shuffle=True, # Shuffle training dataset
            sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            **model_kwargs)

    if args.add_image_time:
        train_loader_image_to_time = DataLoader(
            train_image_to_time_dataset,
            shuffle=True, # Shuffle training dataset
            sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            **model_kwargs)

    train_loaders = {}
    
    train_loaders['image_to_id'] = train_loader_image_to_id

    if args.add_id_id:
        train_loaders['id_to_id'] = train_loader_id_to_id

    if args.add_image_location:
        train_loaders['image_to_location'] = train_loader_image_to_location

    if args.add_image_time:
        train_loaders['image_to_time'] = train_loader_image_to_time

    val_loader = DataLoader(
        val_image_to_id_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True)

    kwargs = {}
    if args.add_image_time:
        kwargs['all_timestamps'] = train_image_to_time_dataset.all_timestamps
    if args.add_image_location:
        kwargs['all_locs'] = train_image_to_location_dataset.all_locs

    model = MKGE(args, num_ent_id, target_list, args.device, **kwargs)

    model.to(args.device)
    
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True, ckpt_path=os.path.join(args.save_dir, 'model.pt'), best_ckpt_path=os.path.join(args.save_dir, 'best_model.pt'))

    params_diff_lr = ['ent_embedding', 'rel_embedding', 'image_embedding', 'location_embedding', 'time_embedding']

    optimizer_grouped_parameters = [
            {"params": [param for p_name, param in model.named_parameters() if not any([x in p_name for x in params_diff_lr])]},
            {"params": model.ent_embedding.parameters(), "lr": args.lr},
            {"params": model.rel_embedding.parameters(), "lr": args.lr},
            {"params": model.image_embedding.parameters(), "lr": args.img_lr},
            {"params": model.location_embedding.parameters(), "lr": args.loc_lr},
            {"params": model.time_embedding.parameters(), "lr": args.time_lr},
        ]

    optimizer = optim.Adam(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay)

    # restore from ckpt
    if args.ckpt_path:
        print('ckpt loaded...')
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['dense_optimizer'])

    for epoch_id in range(args.start_epoch, args.n_epochs):
        print('\nEpoch [%d]:\n' % epoch_id)

        # First run training
        train(model, train_loaders, optimizer, epoch_id, writer, args)

        # Then run val
        val_results, y_pred = evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args)

        if early_stopping.early_stop:
            print("Early stopping...")
            break

    writer.close()

if __name__=='__main__':
    main()
