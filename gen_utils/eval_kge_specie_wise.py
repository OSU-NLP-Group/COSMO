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
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset import iWildCamOTTDataset
from sklearn.metrics import f1_score

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', type=str, default='../iwildcam_v2.0/')
	parser.add_argument('--img-dir', type=str, default='../iwildcam_v2.0/imgs/')
	parser.add_argument('--split', type=str, default='val')
	parser.add_argument('--seed', type=int, default=813765)

	parser.add_argument('--y-pred-path-1', type=str, default=None, help='path to y_pred 1 predictions')
	parser.add_argument('--y-pred-path-2', type=str, default=None, help='path to y_pred 2 predictions')

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

	y_1_pred_dict = json.load(open(args.y_pred_path_1))
	y_2_pred_dict = json.load(open(args.y_pred_path_2))

	total = 0
	train_c = dict([(156, 48007), (1, 10267), (14, 7534), (0, 4078), (5, 4023), (2, 3986), (27, 3584), (54, 3177), (15, 3091), (30, 2740), (31, 2642), (57, 2401), (17, 1966), (12, 1913), (24, 1751), (158, 1709), (160, 1542), (48, 1530), (52, 1444), (32, 1428), (13, 1246), (155, 1168), (33, 1150), (11, 1042), (53, 977), (165, 949), (55, 904), (159, 865), (9, 771), (16, 730), (3, 716), (56, 684), (8, 605), (10, 538), (7, 531), (64, 459), (41, 457), (6, 450), (37, 433), (46, 380), (74, 367), (101, 350), (70, 290), (29, 243), (106, 201), (58, 200), (44, 194), (80, 190), (45, 180), (4, 161), (61, 158), (40, 146), (28, 136), (162, 128), (36, 117), (130, 110), (67, 108), (21, 106), (35, 102), (65, 100), (82, 100), (88, 92), (71, 87), (18, 81), (102, 80), (161, 80), (170, 80), (25, 75), (77, 73), (50, 70), (62, 62), (100, 60), (97, 60), (34, 55), (43, 50), (79, 48), (157, 46), (111, 44), (94, 39), (59, 38), (19, 38), (47, 36), (98, 32), (39, 30), (85, 30), (22, 29), (90, 29), (84, 29), (121, 28), (63, 25), (38, 24), (173, 23), (83, 21), (110, 21), (139, 20), (69, 20), (95, 19), (86, 18), (72, 18), (127, 15), (129, 15), (26, 15), (75, 15), (154, 15), (93, 14), (76, 13), (87, 13), (81, 13), (109, 12), (108, 12), (120, 12), (123, 12), (60, 12), (96, 12), (145, 11), (131, 10), (149, 10), (177, 10), (178, 10), (23, 9), (122, 9), (42, 9), (103, 9), (134, 9), (135, 9), (153, 9), (164, 9), (66, 8), (20, 8), (116, 8), (114, 7), (125, 7), (172, 7), (107, 6), (119, 6), (99, 6), (133, 6), (140, 6), (142, 6), (146, 6), (147, 6), (179, 6), (180, 6), (181, 6), (118, 5), (163, 5), (104, 4), (112, 4), (167, 4), (113, 3), (115, 3), (117, 3), (78, 3), (92, 3), (126, 3), (128, 3), (91, 3), (68, 3), (137, 3), (138, 3), (143, 3), (144, 3), (51, 3), (150, 3), (152, 3), (89, 2), (49, 2), (132, 2), (136, 2), (169, 2), (166, 2), (124, 1), (73, 1), (105, 1), (141, 1), (148, 1), (151, 1), (168, 1), (171, 1), (174, 1), (175, 1), (176, 1)])

	threshold = 100

	acc_1_avg = 0.0
	acc_2_avg = 0.0

	for label_id in y_1_pred_dict:
		# print(type(label_id))

		if train_c[int(label_id)] <= threshold:
			y_1_pred = y_1_pred_dict[label_id]
			y_2_pred = y_2_pred_dict[label_id]

			assert len(y_1_pred)==len(y_2_pred)

			if len(y_1_pred)>0:
				acc_1 = y_1_pred.count(1)*100.0/len(y_1_pred)
				acc_2 = y_2_pred.count(1)*100.0/len(y_1_pred)
				print(f'label_id = {label_id}, acc_1 = {acc_1:.2f}, acc_2 = {acc_2:.2f}, train_count = {train_c[int(label_id)]}')

				acc_1_avg += acc_1
				acc_2_avg += acc_2

				total += 1

	acc_1_avg = acc_1_avg/total
	acc_2_avg = acc_2_avg/total

	print('acc_1_avg = {}'.format(acc_1_avg))
	print('acc_2_avg = {}'.format(acc_2_avg))



