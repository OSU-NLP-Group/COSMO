import pandas as pd
import string
from tqdm import tqdm
import re

ott_taxonomy = pd.read_csv("ott3.3/taxonomy.tsv", sep="\t")
ott_taxonomy = ott_taxonomy.loc[:, ['uid', 'parent_uid', 'name', 'rank', 'sourceinfo', 'uniqname', 'flags']]

punctuation_string = string.punctuation
taxonomy_category = list(ott_taxonomy.name)

for i in tqdm(range(len(taxonomy_category))):
	taxonomy_category[i] = ' '.join(taxonomy_category[i].split())
	taxonomy_category[i] = taxonomy_category[i].translate(str.maketrans('', '', string.punctuation))
	taxonomy_category[i] = taxonomy_category[i].lower()
	taxonomy_category[i] = re.sub(' +', ' ', taxonomy_category[i])

ott_taxonomy_2 = ott_taxonomy.copy()
ott_taxonomy_2.name = taxonomy_category

ott_taxonomy_2.to_csv('ott_taxonomy.csv', index=False)