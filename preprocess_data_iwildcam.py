from wilds import get_dataset
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def gps(x):
    return np.array([x["latitude"], x["longitude"]])

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="iwildcam", download=True)

metadata = pd.read_csv("data/iwildcam_v2.0/metadata.csv")
categories = pd.read_csv("data/iwildcam_v2.0/categories.csv")

# the map is iwildcam_id_to_name {y:name,....}
k = list(categories.y)
v = list(categories.name)

iwildcam_id_to_name = {}
for i in range(len(k)):
    iwildcam_id_to_name[k[i]] = v[i]

iwildcam_name_to_id = {v:k for k,v in iwildcam_id_to_name.items()}

# the map processing (replaces iwildcam category ids by species names)
metadata_y = list(metadata.y)
for i in range(len(metadata_y)):
    metadata_y[i] = iwildcam_id_to_name[metadata_y[i]]

metadata.y = metadata_y
metadata = metadata.loc[:, ["split", "location", "y", "datetime", "filename"]]
metadata.columns = ["split", "location", "name", "datetime", "filename"]

# store time used data
time_used = metadata
# load pre_used_taxonomy.csv to get dic{name:uid}
taxon = pd.read_csv("ott_taxonomy.csv")

k = list(taxon.name)
v = list(taxon.uid)

# the map is taxon_name_to_id {name:uid,....}
taxon_name_to_id = {}
for i in range(len(k)):
    taxon_name_to_id[k[i]] = v[i]

taxon_id_to_name = {x:y for x,y in zip(taxon.uid, taxon.name)}
json.dump(taxon_id_to_name, open('data/iwildcam_v2.0/taxon_id_to_name.json', 'w'), indent=1)

category_offset_non_intersection = max(taxon_name_to_id.values()) + 1


meta_categories = list(set([x for x in metadata.name]))

ott_categories = list(taxon.name)

intersection_categories = list(set(ott_categories) & set(meta_categories))

# intersection of iwildcam and OTT
metadata_intersection = metadata.loc[metadata["name"].isin(intersection_categories), :].copy()

# non-interesection part
metadata_non_intersection = metadata.loc[~metadata["name"].isin(intersection_categories), :].copy()

# replace name by uid in metadata_intersection
metadata_name = list(metadata_intersection.name)
for i in range(len(metadata_name)):
    metadata_name[i] = taxon_name_to_id[metadata_name[i]]

metadata_intersection.name = metadata_name
metadata_intersection.columns = ["split", "location", "uid", "datetime", "filename"]

metadata_non_intersection_name = list(metadata_non_intersection.name)
non_intersection_uids = set()
overall_id_to_name = {}

for i in range(len(metadata_non_intersection_name)):
    specie_name = metadata_non_intersection_name[i]
    metadata_non_intersection_name[i] = iwildcam_name_to_id[specie_name] + category_offset_non_intersection
    non_intersection_uids.add(iwildcam_name_to_id[specie_name])
    overall_id_to_name[metadata_non_intersection_name[i]] = specie_name

metadata_non_intersection.name = metadata_non_intersection_name

intersection_uids = set([iwildcam_name_to_id[taxon_id_to_name[x]] for x in metadata_intersection.uid])

for specie_id in intersection_uids:
    overall_id_to_name[taxon_name_to_id[iwildcam_id_to_name[specie_id]]] = iwildcam_id_to_name[specie_id]

common = non_intersection_uids & intersection_uids
common = [iwildcam_id_to_name[x] for x in common]

json.dump(overall_id_to_name, open('data/iwildcam_v2.0/overall_id_to_name.json', 'w'))

# re-name name column
metadata_non_intersection.columns = ["split", "location", "uid", "datetime", "filename"]

# concatenate metadata_intersection and metadata_non_intersection
metadata = pd.concat([metadata_intersection, metadata_non_intersection])

# store uid used dataset
uid_used = metadata

gps_data = pd.read_json('gps_locations.json')
gps_data = gps_data.transpose()
gps_data.insert(loc=2, column="location", value=gps_data.index.to_list())
gps_data = gps_data.sort_index(ascending=True)

gps_data["GPS"] = gps_data.apply(gps, axis=1)

k = list(gps_data.location)
v = list(gps_data.GPS)

# find the species that have GPS in metadata
metadata = metadata.loc[metadata["location"].isin(k), :].copy()


# the map is dic {location:GPS,....}
dic = {}
for i in range(len(k)):
    dic[k[i]] = v[i]

# make location to GPS
metadata_location = list(metadata.location)
for i in range(len(metadata_location)):
    metadata_location[i] = dic[metadata_location[i]]

metadata.location = metadata_location

# store GPS used data
gps_used = metadata

taxon = taxon.fillna(0)
taxon = taxon.loc[:, ["uid", "parent_uid"]]
taxon.columns = ["h", "t"]
taxon.insert(loc=1, column="r", value=1)
taxon.insert(loc=1, column="datatype_h", value="id")
taxon.insert(loc=4, column="datatype_t", value="id")
taxon.insert(loc=5, column="split", value="train")
taxon.columns = ["h", "datatype_h", "r", "t", "datatype_t", "split"]

takeLocation = gps_used.loc[:, ["filename", "location", "split"]]
takeLocation.insert(loc=1, column="r", value=2)
takeLocation.insert(loc=1, column="datatype_h", value="image")
takeLocation.insert(loc=4, column="datatype_t", value="location")
takeLocation.columns = ["h", "datatype_h", "r", "t", "datatype_t", "split"]

takeTime = time_used.loc[:, ["filename", "datetime", "split"]]
takeTime.insert(loc=1, column="r", value=0)
takeTime.insert(loc=1, column="datatype_h", value="image")
takeTime.insert(loc=4, column="datatype_t", value="time")
takeTime.columns = ["h", "datatype_h", "r", "t", "datatype_t", "split"]

imageIsIn = uid_used.loc[:, ["filename", "uid", "split"]]
imageIsIn.insert(loc=1, column="r", value=3)
imageIsIn.insert(loc=1, column="datatype_h", value="image")
imageIsIn.insert(loc=4, column="datatype_t", value="id")
imageIsIn.columns = ["h", "datatype_h", "r", "t", "datatype_t", "split"]

a = pd.concat([taxon, imageIsIn], ignore_index=True)
a = pd.concat([a, takeTime], ignore_index=True)
a = pd.concat([a, takeLocation], ignore_index=True)

inner = a.loc[(a["datatype_h"]=="image") & (a["datatype_t"]=="id"),:].copy()

ott = a.loc[(a["datatype_h"]=="id") & (a["datatype_t"]=="id"),:].copy()

son = list(ott["h"])
father = list(ott["t"])
paths = {}
for i in tqdm(range(len(son))):
    paths[int(float(son[i]))] = int(float(father[i]))

leaf_node = list(inner.t)
leaf_nodes = []
for item in tqdm(leaf_node):
    if int(float(item)) not in leaf_nodes:
        leaf_nodes.append(int(float(item)))

list_paths = []
def get_paths(leaf_node, paths, nodes_list):
        while leaf_node in  paths.keys():
            # print(leaf_node,"->",paths[leaf_node])
            nodes_list.append(leaf_node)
            leaf_node = paths[leaf_node]

def get_path_nodes(leaf_nodes, paths):
    nodes_list = []
    for item in leaf_nodes:
        get_paths(item,paths,nodes_list)
    return nodes_list

paths_nodes = get_path_nodes(leaf_nodes, paths)

ott["h"] = paths.keys()
ott["t"] = paths.values()

ott = ott.loc[(ott['h'].isin(paths_nodes)) & (ott['t'].isin(paths_nodes)),:]
ott = ott.reset_index()

a = a.loc[(a["datatype_h"] != "id"),:]
a.reset_index()

dataset = pd.concat([ott, a], ignore_index=True)
dataset = dataset.iloc[:,1:]
dataset.to_csv("data/iwildcam_v2.0/dataset_subtree.csv",index = False)
