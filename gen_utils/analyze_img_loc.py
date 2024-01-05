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

def getNumber(x):
    return np.fromstring(x[1:-1], sep=' ', dtype=float)
    
def plot_loc_viz(image_loc_dict, image_loc_dict_val, image_loc_dict_test):
	plt.figure()

	# train
	X_train, Y_train = [], []
	
	for img_filename in tqdm(image_loc_dict):
		x, y = getNumber(image_loc_dict[img_filename])
		X_train.append(x)
		Y_train.append(y)

	scale = 200.0 * np.random.rand(len(X_train))
	plt.subplot(1, 4, 1)
	plt.scatter(X_train, Y_train, label='train', s=scale, alpha=0.3)
	plt.legend()

	# val
	X_val, Y_val = [], []
	
	for img_filename in tqdm(image_loc_dict_val):
		x, y = getNumber(image_loc_dict_val[img_filename])
		X_val.append(x)
		Y_val.append(y)

	scale = 200.0 * np.random.rand(len(X_val))
	plt.subplot(1, 4, 2)
	plt.scatter(X_val, Y_val, label='val', s=scale, alpha=0.3)
	plt.legend()

	# test
	X_test, Y_test = [], []
	
	for img_filename in tqdm(image_loc_dict_test):
		x, y = getNumber(image_loc_dict_test[img_filename])
		X_test.append(x)
		Y_test.append(y)

	scale = 200.0 * np.random.rand(len(X_test))
	plt.subplot(1, 4, 3)
	plt.scatter(X_test, Y_test, label='test', s=scale, alpha=0.3)
	plt.legend()

	plt.subplot(1, 4, 4)
	scale = 200.0 * np.random.rand(len(X_train))
	plt.scatter(X_train, Y_train, label='train', s=scale, alpha=0.3)
	scale = 200.0 * np.random.rand(len(X_val))
	plt.scatter(X_val, Y_val, label='val', s=scale, alpha=0.3)
	scale = 200.0 * np.random.rand(len(X_test))
	plt.scatter(X_test, Y_test, label='test', s=scale, alpha=0.3)
	plt.legend()

	plt.savefig('locs_splits.png')

def plot_loc_hist(n_species_loc):
	# plot histogram
	plt.figure()
	plt.hist(n_species_loc)
	plt.xlabel('No. of species')
	plt.ylabel('No. of locations')
	plt.savefig('n_species_loc_hist.png')

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
	parser.add_argument('--data-dir', type=str, default='iwildcam_v2.0/')
	parser.add_argument('--seed', type=int, default=813765)
	args = parser.parse_args()

	np.random.seed(args.seed)

	mode = 'train'
	datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
	
	datacsv_loc = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'location') & (datacsv['split'] == mode), :]
	datacsv_loc_val = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'location') & (datacsv['split'] == 'val'), :]

	datacsv_id = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'id') & (datacsv['split'] == mode), :]
	datacsv_id_val = datacsv.loc[(datacsv['datatype_h'] == 'image') & (datacsv['datatype_t'] == 'id') & (datacsv['split'] == 'val'), :]

	image_loc_dict, image_loc_dict_val, image_loc_dict_test = {}, {}, {}
	loc_image_dict = defaultdict(list)
	all_locs = set()

	for i in range(len(datacsv_loc)):
		image_filename = datacsv_loc.iloc[i, 0]
		loc = datacsv_loc.iloc[i, -3]
		image_loc_dict[image_filename] = loc
		loc_image_dict[loc].append(image_filename)
		all_locs.add(loc)

	for i in range(len(datacsv_loc_val)):
		image_filename = datacsv_loc_val.iloc[i, 0]
		loc = datacsv_loc_val.iloc[i, -3]
		image_loc_dict_val[image_filename] = loc
	
	all_locs = list(all_locs)

	assert len(image_loc_dict) == len(datacsv_loc)

	all_locs_arr = np.array(list(map(lambda x:getNumber(x), all_locs)))

	(centroid, label, _) = cluster.k_means(all_locs_arr, n_clusters=6)

	centroid_counters = [Counter() for _ in range(6)]

	image_id_dict = create_image_id_dict(datacsv_id)
	image_id_dict_val = create_image_id_dict(datacsv_id_val)

	all_species = list(set(image_id_dict.values()))

	n_species = len(all_species)

	colors = np.random.rand(n_species)

	loc_species_dict = defaultdict(list)

	for img_filename in tqdm(image_loc_dict):
		loc = image_loc_dict[img_filename]
		species_id = image_id_dict[img_filename]

		loc_species_dict[loc].append(species_id)

	n_species_loc = [len(set(loc_species_dict[loc])) for loc in loc_species_dict]

	n_avg_species_loc = np.average(n_species_loc)

	for loc in tqdm(loc_species_dict):
		centroid_counters[label[all_locs.index(loc)]].update(loc_species_dict[loc])

	# plot locations of train/val/test
	plot_loc_viz(image_loc_dict, image_loc_dict_val, image_loc_dict_test)

	centroid_counters_val = [Counter() for _ in range(6)]

	for img_filename in tqdm(image_loc_dict_val):
		loc = getNumber(image_loc_dict_val[img_filename])

		# find closest of (x, y) to each of train's centroid points
		loc_dist = np.linalg.norm(loc - centroid, axis=-1)
		cluster_id = np.argmin(loc_dist)

		# assign centroid label to this point
		species_id = image_id_dict_val[img_filename]
		centroid_counters_val[cluster_id].update([species_id])


	confus_mat = np.ones((6, 6))*np.inf

	for i in range(6):
		for j in range(6):
			if len(centroid_counters[i])>0 and len(centroid_counters_val[j])>0:
				confus_mat[i, j] = calc_dist_train_val(centroid_counters, centroid_counters_val, i, j)


	fig = plt.figure()
	cax = plt.matshow(confus_mat)
	fig.colorbar(cax)
	plt.savefig('loc_corr_analysis.png')
	



