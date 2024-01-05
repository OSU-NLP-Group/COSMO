import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import re
from tqdm import tqdm
import sklearn.cluster as cluster
from collections import defaultdict, Counter
from re import match

def getSeparated(item):
    m = match(r"(.*)-(.*)-(.*) (.*):(.*):(\d{2})", item)
    years, month, day, hour, minutes, second = m.groups()
    return int(years), int(month), int(day), int(hour), int(minutes), int(second)

def create_image_id_dict(datacsv_id):
	image_id_dict = {}

	for i in range(len(datacsv_id)):
		image_filename = datacsv_id.iloc[i, 0]
		specie_id = int(float(datacsv_id.iloc[i, -3]))

		image_id_dict[image_filename] = specie_id

	return image_id_dict

def bhattacharyya_distance(distribution1, distribution2):
    """ Estimate Bhattacharyya Distance (between General Distributions)
    
    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    sq = 0
    for i in range(len(distribution1)):
        sq  += np.sqrt(distribution1[i]*distribution2[i])
    
    return -np.log(sq)

def calc_dist_train_val(centroid_counters, centroid_counters_val, idx1, idx2):
	counter_0_keys = list(set(centroid_counters[idx1].keys()) | set(centroid_counters_val[idx2].keys()))

	counter_0_train_dist, counter_0_val_dist = np.zeros(len(counter_0_keys)), np.zeros(len(counter_0_keys))

	for item in centroid_counters[idx1]:
		counter_0_train_dist[counter_0_keys.index(item)] += centroid_counters[idx1][item]
	counter_0_train_dist = counter_0_train_dist/np.sum(counter_0_train_dist)

	for item in centroid_counters_val[idx2]:
		counter_0_val_dist[counter_0_keys.index(item)] += centroid_counters_val[idx2][item]
	counter_0_val_dist = counter_0_val_dist/np.sum(counter_0_val_dist)

	counter_0_dist = bhattacharyya_distance(counter_0_train_dist, counter_0_val_dist)
	return counter_0_dist

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', type=str, default='../iwildcam_v2.0/')
	parser.add_argument('--seed', type=int, default=813765)
	args = parser.parse_args()

	np.random.seed(args.seed)

	datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
	
	datacsv_time_train = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'time') & (datacsv['split'] == 'train'), :]
	datacsv_time_val = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'time') & (datacsv['split'] == 'val'), :]


	datacsv_id_train = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'id') & (datacsv['split'] == 'train'), :]
	datacsv_id_val = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'id') & (datacsv['split'] == 'val'), :]
	
	# compute distribution over hh, mm, ss
	image_time_dict_train, image_time_dict_val = {}, {}

	for i in range(len(datacsv_time_train)):
		image_filename = datacsv_time_train.iloc[i, 0]
		time = datacsv_time_train.iloc[i, -3]
		image_time_dict_train[image_filename] = time

	for i in range(len(datacsv_time_val)):
		image_filename = datacsv_time_val.iloc[i, 0]
		time = datacsv_time_val.iloc[i, -3]
		image_time_dict_val[image_filename] = time

	image_id_dict_train = create_image_id_dict(datacsv_id_train)
	image_id_dict_val = create_image_id_dict(datacsv_id_val)

	# calculate confus matrix for different hours
	species_c_hour_train = [Counter() for _ in range(24)]

	for img_filename in tqdm(image_time_dict_train):
		time = image_time_dict_train[img_filename]
		yyyy, mon, dd, hh, mm, ss = getSeparated(time)

		species_id = image_id_dict_train[img_filename]
		species_c_hour_train[hh].update([species_id])

	species_c_hour_val = [Counter() for _ in range(24)]

	for img_filename in tqdm(image_time_dict_val):
		time = image_time_dict_val[img_filename]
		yyyy, mon, dd, hh, mm, ss = getSeparated(time)

		species_id = image_id_dict_val[img_filename]
		species_c_hour_val[hh].update([species_id])

	confus_mat = np.ones((24, 24))*np.inf

	for i in range(24):
		for j in range(24):
			if len(species_c_hour_train[i])>0 and len(species_c_hour_val[j])>0:
				confus_mat[i, j] = calc_dist_train_val(species_c_hour_train, species_c_hour_val, i, j)


	fig = plt.figure()
	cax = plt.matshow(confus_mat)
	fig.colorbar(cax)
	plt.savefig('time_corr_analysis.png')

	

