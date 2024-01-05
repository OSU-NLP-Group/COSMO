import os
import torch
import numpy as np
import re
from math import pi
from re import match
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import pandas as pd

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

class iWildCamOTTDataset(Dataset):
    def __init__(self, datacsv, mode, args, entity2id, target_list, head_type=None, tail_type=None):
        super(iWildCamOTTDataset, self).__init__()
        if head_type is not None and tail_type is not None:
            self.datacsv = datacsv.loc[(datacsv['datatype_h'] == head_type) & (datacsv['datatype_t'] == tail_type) & (
                    datacsv['split'] == mode), :]
            print("length of {}2{} dataset = {}".format(head_type, tail_type, len(self.datacsv)))
        else:
            self.datacsv = datacsv.loc[datacsv['split'] == mode, :]
            print("length of alltype dataset = {}".format(len(self.datacsv)))
        self.args = args
        self.mode = mode
        self.entity2id = entity2id
        self.target_list = target_list
        self.entity_to_species_id = {self.target_list[i, 0].item():i for i in range(len(self.target_list))}

        # print(self.entity_to_species_id)

        if args.use_data_subset:
            train_indices = np.random.choice(np.arange(len(self.datacsv)), size=args.subset_size, replace=False)
            self.datacsv = self.datacsv.iloc[train_indices]
        
        datacsv_loc = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'location')]

        self.location_to_id = {}

        for i in range(len(datacsv_loc)):
            loc = datacsv_loc.iloc[i, -3]

            assert loc[0] == '['
            assert loc[-1] == ']'
            # print(loc)
            if loc not in self.location_to_id:
                self.location_to_id[loc] = len(self.location_to_id)

        self.all_timestamps = None

        if head_type == 'image' and tail_type == 'time':
            datacsv_time = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'time')]
            self.time_to_id = {}

            for i in range(len(datacsv_time)):
                time = datacsv_time.iloc[i, -3]

                _, hour = get_separate_time(time)

                _HOUR_RAD = 2 * pi / 24

                h1, h2 = point(hour, _HOUR_RAD)

                time = hour

                if time not in self.time_to_id:
                    self.time_to_id[time] = len(self.time_to_id)

            self.all_timestamps = torch.stack(list(map(lambda x:torch.tensor(x), self.time_to_id.keys())))
            if len(self.all_timestamps.size())==1:
                self.all_timestamps = self.all_timestamps.unsqueeze(-1)

        self.all_locs = torch.stack(list(map(lambda x:getNumber(x), self.location_to_id.keys())))

    def __len__(self):
        return len(self.datacsv)

    def __getitem__(self, idx):

        head_type = self.datacsv.iloc[idx, 1]
        tail_type = self.datacsv.iloc[idx, -2]
        head = self.datacsv.iloc[idx, 0]
        relation = self.datacsv.iloc[idx, 2]
        tail = self.datacsv.iloc[idx, -3]

        # for tail extract
        h = None
        t = None

        if tail_type == "id":
            if head_type in ["image", "location"]:
                t = torch.tensor([self.entity_to_species_id[self.entity2id[str(int(float(tail)))]]], dtype=torch.long).squeeze(-1)
            else:
                t = torch.tensor([self.entity2id[str(int(float(tail)))]], dtype=torch.long).squeeze(-1)

        elif tail_type == "location":
            t = self.location_to_id[tail]

        elif tail_type == "time":
            tail = datatime_divide(tail, self.args)
            t = self.time_to_id[tail]

        # for head extract
        if head_type == "id":
            h = torch.tensor([self.entity2id[str(int(float(head)))]], dtype=torch.long).squeeze(-1)

        elif head_type == "image":
            img = Image.open(os.path.join(self.args.img_dir, head)).convert('RGB')
            
            transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
            h = transform_steps(img)
        
        elif head_type == "location":
            h = getNumber(head)

        # for r extract
        r = torch.tensor([int(relation)])

        return h, r, t

def getNumber(x):
    return torch.tensor(np.fromstring(x[1:-1], dtype=float, sep=' '), dtype=torch.float)

def get_separate_time(item):
    m = match(r"(.*)-(.*)-(.*) (.*):(.*):(\d{2})", item)
    years, month, day, hour, minutes, second = m.groups()
    return float(month), float(hour)

def datatime_divide(timestamp, args):
    month, hour = get_separate_time(timestamp)
    
    _HOUR_RAD = 2 * pi / 24

    h1, h2 = point(hour, _HOUR_RAD)

    return hour
        
def point(m, rad):
    from math import sin, cos
    # place on circle
    return sin(m * rad), cos(m * rad)


