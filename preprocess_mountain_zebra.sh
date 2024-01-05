#!/bin/bash

wget https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip
mkdir inaturalist-taxonomy.dwca/
unzip -d inaturalist-taxonomy.dwca/ inaturalist-taxonomy.dwca.zip

python process_taxonomy_inat.py

mv data/snapshot_mountain_zebra/MTZ_public/MTZ_S1 data/snapshot_mountain_zebra/

python preprocess_data_mountain_zebra.py --data-dir data/snapshot_mountain_zebra/ --dataset-prefix mountain_zebra --species-common-names-file data/snapshot_mountain_zebra/category_to_label_map.json
