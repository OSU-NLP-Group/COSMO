import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict
import math
import torchvision.transforms as transforms

from resnet import Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy
from PIL import Image
from pytorchtools import EarlyStopping

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def run_epoch(model, train_loader, val_loader, optimizer, epoch, args, early_stopping, train):

    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []

    batches = train_loader if train else val_loader

    batches = tqdm(batches)
    last_batch_idx = len(batches)-1

    criterion = nn.CrossEntropyLoss()

    batch_idx = 0
    for labeled_batch in batches:
        if train:
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

            # compute objective
            loss = criterion(batch_results['y_pred'], batch_results['y_true'])
            batch_results['objective'] = loss.item()
            loss.backward()

            # update model and logs based on effective batch
            optimizer.step()
            model.zero_grad()

        else:
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

            batch_results['objective'] = criterion(batch_results['y_pred'], batch_results['y_true']).item()
        
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)

        effective_batch_idx = batch_idx + 1

        batch_idx += 1
        if args.debug and batch_idx > 100:
            break

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    # epoch_metadata = collate_list(epoch_metadata)

    metrics = [
        Accuracy(prediction_fn=None),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    results_str = (
        f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
    )
    
    if not train: # just for eval.
        early_stopping(-1*results[metrics[0].agg_metric_field], model, optimizer)

    results['epoch'] = epoch
    # if dataset['verbose']:
    print('Epoch eval:\n')
    print(results_str)

    return results, epoch_y_pred

class iWildCamDataset(Dataset):
    def __init__(self, datacsv, img_dir, mode, entity2id, target_list):  # dic_data <- datas
        super(iWildCamDataset, self).__init__()
        self.mode = mode
        self.datacsv = datacsv.loc[datacsv['split'] == mode, :]
        self.img_dir = img_dir
        self.entity2id = entity2id
        self.target_list = target_list
        self.entity_to_species_id = {self.target_list[i, 0].item():i for i in range(len(self.target_list))}

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):
        y = torch.tensor([self.entity_to_species_id[self.entity2id[str(int(float(self.datacsv.iloc[idx, 3])))]]], dtype=torch.long).squeeze()

        img = Image.open(os.path.join(self.img_dir, self.datacsv.iloc[idx, 0])).convert('RGB')

        transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
        x = transform_steps(img)

        return x, y

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
    print("No. of target categories = {}".format(len(categories)))
    return torch.tensor(categories, dtype=torch.long).unsqueeze(-1)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['iwildcam', 'mountain_zebra'], default='iwildcam')
    parser.add_argument('--data-dir', type=str, default='iwildcam_v2.0/')
    parser.add_argument('--img-dir', type=str, default='iwildcam_v2.0/imgs/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=813765)
    parser.add_argument('--save-dir', type=str, default='ckpts/toy/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='early stop if metric does not improve for x epochs')
    
    args = parser.parse_args()

    print('args = {}'.format(args))

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(args.device) > device_count:
            raise ValueError(f"Specified {len(args.device)} devices, but only {device_count} devices found.")

        device_str = ",".join(map(str, args.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # Set random seed
    # set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'iwildcam':
        datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
    else:
        datacsv = pd.read_csv(os.path.join(args.data_dir, 'data_triples.csv'), low_memory=False)

    datacsv = datacsv.loc[(datacsv["datatype_h"] == "image") & (datacsv["datatype_t"] == "id")]

    entity2id = {} # each of triple types have their own entity2id
        
    for i in tqdm(range(datacsv.shape[0])):
        _get_id(entity2id, str(int(float(datacsv.iloc[i,3]))))

    print('len(entity2id) = {}'.format(len(entity2id)))

    target_list = generate_target_list(datacsv, entity2id)

    train_dataset = iWildCamDataset(datacsv, args.img_dir, 'train', entity2id, target_list)
    val_dataset = iWildCamDataset(datacsv, args.img_dir, 'val', entity2id, target_list)

   
    train_loader = DataLoader(
        train_dataset,
        shuffle=True, # Shuffle training dataset
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        shuffle=False, # Do not shuffle eval datasets
        sampler=None,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True)

    model = Resnet50(args)
    model.to(args.device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay)

    best_val_metric = None
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True, ckpt_path=os.path.join(args.save_dir, 'model.pt'), best_ckpt_path=os.path.join(args.save_dir, 'best_model.pt'))

    for epoch in range(args.n_epochs):
        print('\nEpoch [%d]:\n' % epoch)

        # First run training
        run_epoch(model, train_loader, val_loader, optimizer, epoch, args, early_stopping, train=True)

        # Then run val
        val_results, y_pred = run_epoch(model, train_loader, val_loader, optimizer, epoch, args, early_stopping, train=False)

        if early_stopping.early_stop:
            print("Early stopping...")
            break
    
if __name__=='__main__':
    main()
