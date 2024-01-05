import pandas as pd
import string
from tqdm import tqdm
import re
import json

inat_taxonomy = pd.read_csv("inaturalist-taxonomy.dwca/taxa.csv")
inat_taxonomy.fillna('')

inat_taxonomy = inat_taxonomy.loc[:, ['id', 'parentNameUsageID', 'scientificName', 'taxonRank']]
inat_taxonomy.columns = ['uid', 'parent_uid', 'name', 'taxonRank']

punctuation_string = string.punctuation
taxonomy_category = list(inat_taxonomy.name)

for i in tqdm(range(len(taxonomy_category))):
	taxonomy_category[i] = ' '.join(taxonomy_category[i].split())
	taxonomy_category[i] = taxonomy_category[i].translate(str.maketrans('', '', string.punctuation))
	taxonomy_category[i] = taxonomy_category[i].lower()
	taxonomy_category[i] = re.sub(' +', ' ', taxonomy_category[i])

inat_taxonomy_2 = inat_taxonomy.copy()
inat_taxonomy_2.name = taxonomy_category

# replace all parent ids by just ids
parent_uids = list(inat_taxonomy_2.parent_uid)

parent_uids_new = []

for x in parent_uids:
	if isinstance(x, str):
		parent_uids_new.append(x.replace('https://www.inaturalist.org/taxa/',''))
	else:
		parent_uids_new.append('')

inat_taxonomy_2.parent_uid = parent_uids_new

inat_taxonomy_2 = inat_taxonomy_2.loc[inat_taxonomy_2['parent_uid'] != '']

taxon = inat_taxonomy_2
taxon = taxon.fillna(0)
taxon = taxon.loc[:, ["uid", "parent_uid"]]
taxon.columns = ["h", "t"]
taxon.insert(loc=1, column="r", value=1)
taxon.insert(loc=1, column="datatype_h", value="id")
taxon.insert(loc=4, column="datatype_t", value="id")
taxon.insert(loc=5, column="split", value="train")
taxon.columns = ["h", "datatype_h", "r", "t", "datatype_t", "split"]

son = list(taxon["h"])
father = list(taxon["t"])
paths = {}

for i in tqdm(range(len(son))):
	if isinstance(father[i], str) and len(father[i])==0:
		print('flag 1')
		continue

	paths[int(float(son[i]))] = int(float(father[i]))
	
taxon_id_to_name = json.load(open('data/snapshot_mountain_zebra/taxon_id_to_name_lila.json'))
category_to_label_map = json.load(open('data/snapshot_mountain_zebra/category_to_label_map_lila.json'))
taxon_name_to_id = {v:k for k,v in taxon_id_to_name.items()}

category_names = []

for x in tqdm(category_to_label_map):
    if category_to_label_map[x] in taxon_name_to_id:
        category_names.append(taxon_name_to_id[category_to_label_map[x]])
    else:
        print(category_to_label_map[x])

leaf_node = category_names
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

taxon["h"] = paths.keys()
taxon["t"] = paths.values()

taxon = taxon.loc[(taxon['h'].isin(paths_nodes)) & (taxon['t'].isin(paths_nodes)),:]
# taxon = taxon.reset_index()

print('len(taxon) = {}'.format(len(taxon)))

out_file = 'data/snapshot_mountain_zebra/taxon.csv'
taxon.to_csv(out_file, index=False)
