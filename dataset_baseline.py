import os
import h5py
import torch
import numpy as np
import re
from math import pi
from re import match
from PIL import Image
from torchvision import transforms
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import pandas as pd

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

class iWildCamOTTDataset(Dataset):
    def __init__(self, datacsv, mode, args, entity2id, target_list, disjoint=True, output_subgraph=False, is_train=False):  # dic_data <- datas
        super(iWildCamOTTDataset, self).__init__()

        self.datacsv_id = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'id') & (datacsv['split'] == mode), :]
        self.datacsv_loc = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'location') & (datacsv['split'] == mode), :]
        self.datacsv_time = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'time') & (datacsv['split'] == mode), :]

        # create dataframe with both location and time
        self.datacsv_loc_time_left = pd.merge(self.datacsv_loc, self.datacsv_time, how='left', left_on=['h','datatype_h','split'], right_on=['h','datatype_h','split'])
        loc = torch.stack([getNumber(x) for x in self.datacsv_loc.loc[:, 't'].values.tolist()], dim=0)

        # print(loc)
        # print('loc = {}'.format(loc.size()))
        self.loc_avg = loc.mean(dim=0)

        if args.dataset == 'iwildcam':
            time = torch.stack([torch.tensor(datatime_divide(x, args)) for x in self.datacsv_time.loc[:, 't'].values.tolist()])
        else:
            time = torch.stack([torch.tensor(date_divide(x, args)) for x in self.datacsv_time.loc[:, 't'].values.tolist()])

        # print(time)
        self.time_avg = time.mean(dim=0)

        # print('time = {}'.format(time.size()))

        # print('self.loc_avg = {}'.format(self.loc_avg))
        # print('self.time_avg = {}'.format(self.time_avg))

        self.datacsv_loc_time = pd.merge(self.datacsv_loc, self.datacsv_time, how='outer', left_on=['h','datatype_h','split'], right_on=['h','datatype_h','split'])

        # r remains id 2 (corr. to location)
        # h,datatype_h,r,t,datatype_t,split
        self.datacsv_loc_time = self.datacsv_loc_time.loc[:, ['h','t_x','t_y','split']]
        self.datacsv_loc_time.columns = ['h','location', 'time','split']
        
        datacsv_ilt = pd.merge(self.datacsv_loc_time, self.datacsv_id, how='outer', left_on=['h','split'], right_on=['h','split'])
        datacsv_ilt = datacsv_ilt.loc[:, ['h','location', 'time', 't', 'split']]
        datacsv_ilt.columns = ['image','location', 'time', 'species_id', 'split']

        # print(len(self.datacsv))
        # print(self.datacsv.head())

        self.datacsv = datacsv_ilt

        # print("The length of {}2{} dataset is {}".format(head_type, tail_type, len(self.datacsv)))

        self.args = args
        self.mode = mode
        self.entity2id = entity2id
        self.target_list = target_list
        self.entity_to_species_id = {self.target_list[i, 0].item():i for i in range(len(self.target_list))}

        # print(self.entity_to_species_id)

        if args.use_data_subset:
            train_indices = np.random.choice(np.arange(len(self.datacsv)), size=args.subset_size, replace=False)
            self.datacsv = self.datacsv.iloc[train_indices]
        
        
        # print('shape(self.datacsv) = {}'.format(self.datacsv.shape))
        # print(self.datacsv.head())
        datacsv_loc = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'location')]


        self.location_to_id = {}
        # print(datacsv_loc)

        if args.dataset == 'iwildcam':    
            for i in range(len(datacsv_loc)):
                loc = datacsv_loc.iloc[i, 3]

                assert loc[0] == '['
                assert loc[-1] == ']'
                # print(loc)
                if self.args.use_cluster_centroids_for_location:
                    loc = self.loc_centroid_map[loc]

                if loc not in self.location_to_id:
                    self.location_to_id[loc] = len(self.location_to_id)

            if self.args.use_location_breakdown:
                self.all_locs = torch.stack(list(map(lambda x:GPSToHMS(x), self.location_to_id.keys())))
            else:            
                self.all_locs = torch.stack(list(map(lambda x:getNumber(x), self.location_to_id.keys())))

        # print(self.location_to_id)
        self.all_timestamps = None

        datacsv_time = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'time')]
        self.time_to_id = {}

        for i in range(len(datacsv_time)):
            time = datacsv_time.iloc[i, 3]

            if self.args.dataset == 'iwildcam':
                month, hour = get_separate_time(time)
            else:
                # month = get_separate_date(time)
                # print(time)
                month, hour = get_separate_time(time)

            _HOUR_RAD = 2 * pi / 24
            _MONTH_RAD = 2 * pi / 12

            m1, m2 = point(month, _MONTH_RAD)

            if self.args.dataset == 'iwildcam':
                h1, h2 = point(hour, _HOUR_RAD)

            if self.args.only_hour:
                if self.args.use_circular_space:
                    time = (h1, h2)
                else:
                    time = (hour,)
            elif self.args.only_month or self.args.dataset in ['inat18', 'inat21_mammals']:
                if self.args.use_circular_space:
                    time = (m1, m2)
                else:
                    time = (month,)
            else:
                if self.args.use_circular_space:
                    time = (m1, m2, h1, h2)
                else:
                    time = (month, hour)

            if time not in self.time_to_id:
                self.time_to_id[time] = len(self.time_to_id)

        # print(self.time_to_id)
        self.all_timestamps = torch.stack(list(map(lambda x:torch.tensor(x), self.time_to_id.keys())))
        if len(self.all_timestamps.size())==1:
            self.all_timestamps = self.all_timestamps.unsqueeze(-1)

        # print('all_timestamps = {}'.format(self.all_timestamps.size()))

        if self.args.img_embed_model in ['resnet18', 'resnet50']:
            self.transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.datacsv)

    # @profile
    def __getitem__(self, idx):
        # 'image','location', 'time', 'species_id', 'split'

        image_filename = self.datacsv.iloc[idx, 0]
        img = Image.open(os.path.join(self.args.img_dir, image_filename)).convert('RGB')
        # transform_steps = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize(_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN, _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)])
        img = self.transform_steps(img)

        edge_index, edge_type = [], []

        location = self.datacsv.iloc[idx, 1]

        location_inp = None
        if isinstance(location, float) and np.isnan(location):
            location_inp = self.loc_avg

        time = self.datacsv.iloc[idx, 2]

        time_inp = None
        if isinstance(time, float) and np.isnan(time):
            time_inp = self.time_avg

        species_id = self.datacsv.iloc[idx, 3]
        species_id = torch.tensor([self.entity_to_species_id[self.entity2id[str(int(float(species_id)))]]], dtype=torch.long).squeeze(-1)

        if location_inp is None:
            location_inp = getNumber(location)
        
        if time_inp is None:
            if self.args.dataset == 'iwildcam':
                time_inp = torch.tensor(datatime_divide(time, self.args))
            else:
                time_inp = torch.tensor(date_divide(time, self.args))
            
        return img, location_inp, time_inp, species_id
        

def getNumber(x):
    # return torch.tensor(np.array(re.findall(r"\d+\.?\d*", x), dtype=float), dtype=torch.float)
    return torch.tensor(np.fromstring(x[1:-1], dtype=float, sep=' '), dtype=torch.float)

def get_separate_time(item):
    m = match(r"(.*)-(.*)-(.*) (.*):(.*):(\d{2})", item)
    years, month, day, hour, minutes, second = m.groups()
    return float(month), float(hour)

def get_separate_date(item):
    m = match(r"(.*)-(.*)-(.*)", item)
    years, month, day = m.groups()
    return float(month)

def datatime_divide(timestamp, args):    # season{0:spring, 1: summer 2:autumn, 3:winter}   hor{0:day, 1:night}
    month, hour = get_separate_time(timestamp)
    
    _HOUR_RAD = 2 * pi / 24
    _MONTH_RAD = 2 * pi / 12

    m1, m2 = point(month, _MONTH_RAD)
    h1, h2 = point(hour, _HOUR_RAD)

    if args.only_hour:
        if args.use_circular_space:
            return (h1, h2)
        else:
            return (hour,)
    elif args.only_month:
        if args.use_circular_space:
            return (m1, m2)
        else:
            return (month,)

    # if hour < 5 or hour > 18:
    #     day_night = 0
    # else:
    #     day_night = 1
    # print('timestamp = {}, day_night = {}'.format(timestamp, day_night))
    if args.use_circular_space:
        return (m1, m2, h1, h2)
    else:
        return (month, hour)

def date_divide(timestamp, args):
    month = get_separate_date(timestamp)
    
    _MONTH_RAD = 2 * pi / 12

    m1, m2 = point(month, _MONTH_RAD)

    if args.use_circular_space:
        return (m1, m2)
    else:
        return (month,)

def point(m, rad):
    from math import sin, cos
    # place on circle
    return sin(m * rad), cos(m * rad)


def separate(year):
    regex = "^(?P<century>\d{0,2}?)(?P<decade>\d?)(?P<year>\d)$"
    return match(regex, year)


def getSeparated(item):
    _MINUTE_RAD = 2 * pi / 60
    _HOUR_RAD = 2 * pi / 24
    _DAY_RAD = 2 * pi / 31
    _MONTH_RAD = 2 * pi / 12
    _YEAR_DECADE_RAD = 2 * pi / 10
    m = match(r"(.*)-(.*)-(.*) (.*):(.*):(\d{2})", item)
    years, month, day, hour, minutes, second = m.groups()
    separated = separate(years)
    c = int(separated.group('century'))
    decade = int(separated.group('decade'))
    year = int(separated.group('year'))
    dec1, dec2 = point(decade, _YEAR_DECADE_RAD)
    y1, y2 = point(year, _YEAR_DECADE_RAD)
    m1, m2 = point(int(month), _MONTH_RAD)
    d1, d2 = point(int(day), _DAY_RAD)
    h1, h2 = point(int(hour), _HOUR_RAD)
    min1, min2 = point(int(minutes), _MINUTE_RAD)
    sec1, sec2 = point(int(second), _MINUTE_RAD)
    return torch.tensor(np.array([c, dec1, dec2, y1, y2, m1, m2, d1, d2, h1, h2, min1, min2, sec1, sec2]),
                        dtype=torch.float)

def D2Dms(d_data):
    d_data = float(d_data)
    d = int(d_data)
    m = int((d_data-d)*60)
    s = ((d_data-d)*60-m)*60
    return d,m,s


def GPSToHMS(x, parse_regex=True):
    # print('x = {}'.format(x))

    if parse_regex:
        a = re.findall(r"\d+\.?\d*", x)
    else:
        a = x

    # print('a = {}'.format(a))
    lon = a[0]
    lat = a[1]
    # print('lat = {}'.format(lat))
    # print('lon = {}'.format(lon))

    dl, ml, sl = D2Dms(lon)
    da, ma, sa = D2Dms(lat)

    # print(f'dl = {dl}, ml = {ml}, sl = {sl}')
    # print(f'da = {da}, ma = {ma}, sa = {sa}')

    _MINUTE_RAD = 2 * pi / 60
    _HOUR_RAD = 2 * pi / 24

    dl_1, dl_2 = point(int(dl), _HOUR_RAD)
    ml_1, ml_2 = point(int(ml), _MINUTE_RAD)
    sl_1, sl_2 = point(int(sl), _MINUTE_RAD)

    da_1, da_2 = point(int(da), _HOUR_RAD)
    ma_1, ma_2 = point(int(ma), _MINUTE_RAD)
    sa_1, sa_2 = point(int(sa), _MINUTE_RAD)

    # print(f'dl_1 = {dl_1}, dl_2 = {dl_2}, ml_1 = {ml_1}, ml_2 = {ml_2}, sl_1 = {sl_1}, sl_2 = {sl_2}')
    # print(f'da_1 = {da_1}, da_2 = {da_2}, ma_1 = {ma_1}, ma_2 = {ma_2}, sa_1 = {sa_1}, sa_2 = {sa_2}')

    return torch.tensor(np.array([dl_1, dl_2, ml_1, ml_2, sl_1, sl_2, da_1, da_2, ma_1, ma_2, sa_1, sa_2], dtype=float), dtype=torch.float)

