import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
import random, string
import os


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/snapshot_mountain_zebra/')
    parser.add_argument('--use-loc-canonical-id', action='store_true')
    parser.add_argument('--no-drop-nonexist-imgs', action='store_true')
    parser.add_argument('--split-dataset', action='store_true', help='randomly split into train/val/test splits')
    parser.add_argument('--no-datetime', action='store_true', help='ignore date/time')
    parser.add_argument('--no-location', action='store_true', help='ignore location')
    parser.add_argument('--species-common-names-file', type=str, default='data/snapshot_mountain_zebra/category_to_label_map.json')
    parser.add_argument('--img-prefix', type=str, default='')
    parser.add_argument('--dataset-prefix', type=str, default='')
    
    
    args = parser.parse_args()

    annot_file = os.path.join(args.data_dir, 'annotations.json')
    loc_file = os.path.join(args.data_dir, 'locations.csv')
    category_to_label_map = json.load(open(args.species_common_names_file, 'r'))

    annotations_json = json.load(open(annot_file))

    taxon_id_to_name_filename = 'snapshot_mountain_zebra/taxon_id_to_name_lila.json'

    taxon_id_to_name = json.load(open(taxon_id_to_name_filename, 'r'))
    taxon_name_to_id = {v:k for k,v in taxon_id_to_name.items()}

    print('len(taxon_name_to_id) = {}'.format(len(taxon_name_to_id)))
    print('len(taxon_id_to_name) = {}'.format(len(taxon_id_to_name)))

    if os.path.exists(loc_file) and not args.use_loc_canonical_id:
        location_coordinates = pd.read_csv(loc_file)
    else:
        location_coordinates = None

    img_json = annotations_json['images']
    img_json = [x for x in img_json if args.no_drop_nonexist_imgs or os.path.exists(os.path.join(args.data_dir, args.img_prefix, x['file_name']))]
    # print(img_json[0].keys())

    # add y labels to metadata
    annotations = annotations_json['annotations']

    annotations_image_id = [x['image_id'] for x in annotations]
    annotations_category_id = [x['category_id'] for x in annotations]

    annotations_df = pd.DataFrame(list(zip(annotations_image_id, annotations_category_id)), columns=['image_id', 'category_id'])

    metadata = annotations_df

    if 'caltech' in args.data_dir:
        datetime_field = 'date_captured'
    else:
        datetime_field = 'datetime'

    # add image filename
    img_ids = [x['id'] for x in tqdm(img_json)]
    img_filenames = [(args.img_prefix + x['file_name']) for x in img_json]

    img_loc = [x['location'] for x in img_json]

    if not args.no_datetime:
        img_datetime = [x[datetime_field] for x in img_json]
        img_df = pd.DataFrame(list(zip(img_ids, img_filenames, img_loc, img_datetime)), columns=['image_id', 'filename', 'location', 'datetime'])
    else:
        img_df = pd.DataFrame(list(zip(img_ids, img_filenames, img_loc)), columns=['image_id', 'filename', 'location'])

    # construct a df with location paired to split

    # TODO; check if list of locations is in order
    locs = list(img_df.location)
    splits = []

    split_json_file = open(os.path.join(args.data_dir, 'splits.json'))
    split_json = json.load(split_json_file)

    train_locs = set(split_json['splits']['train'])
    val_locs = set(split_json['splits']['val'])

    if 'test' in split_json['splits']:
        test_locs = set(split_json['splits']['test'])
    else:
        test_locs = set()
    
    for loc in locs:
        if loc in train_locs:
            splits.append('train')
        elif loc in val_locs:
            splits.append('val')
        elif loc in test_locs:
            splits.append('test')

    print('len(img_df) = {}'.format(len(img_df)))
    print('len(splits) = {}'.format(len(splits)))

    img_df = img_df.assign(split=splits)
    # print(img_df.head())
    print(img_df[img_df['split']=='val'])
    print(img_df[img_df['split']=='test'])

    img_df = img_df.drop_duplicates(subset=['image_id'])

    if location_coordinates is not None:
        location_coordinates.columns = ['location', 'elevation', 'geometry']
        img_df = pd.merge(img_df, location_coordinates, how='left', left_on=['location'], right_on=['location']) # [location', 'date', 'image_id', 'category_id', 'filename']

        # replace location by actual (lat, lon) coordinates

        locs = [img_df.iloc[i, -1] for i in range(len(img_df))]
        locs = [np.array(x.replace('c(','').replace(')','').split(', ')).astype(float) for x in locs]
        # print(locs)

        img_df.location = locs

        # print(img_df.head())
        # print(img_df.columns)
    elif not args.no_location:
        locs = list(img_df.location)
        locs = ['{}_{}'.format(loc, args.dataset_prefix) for loc in locs]
        img_df.location = locs


    metadata = metadata.drop_duplicates(subset=['image_id'], keep=False)

    # print duplicates
    # ids = metadata['image_id']
    # print(metadata[ids.isin(ids[ids.duplicated()])].sort_values('image_id'))

    print('len(img_df) = {}'.format(len(img_df)))
    print('len(metadata) before = {}'.format(len(metadata)))

    metadata = pd.merge(metadata, img_df, how='inner', left_on=['image_id'], right_on=['image_id']) # [location', 'date', 'image_id', 'category_id', 'filename']
    print(metadata.columns)
    print(metadata.head())

    print('len(metadata) after = {}'.format(len(metadata)))

    # add category names
    category = annotations_json['categories']
    # print('species_labels = {}'.format(species_labels))
    print('len(category) before = {}'.format(len(category)))
    # print(category)

    if 'ena24' in args.data_dir:
        for item in category:
            item['name'] = item['name'].lower()

    category = [x for x in category if x['name'] in category_to_label_map]

    print('len(category) after = {}'.format(len(category)))
    # print('category after = {}'.format([x['name'] for x in category]))

    category_ids = [x['id'] for x in category]
    category_names = [taxon_name_to_id[category_to_label_map[x['name']]] for x in category]

    # for x in category_names:
        # print(x in all_taxons)
        # assert x in all_taxons

    category_df = pd.DataFrame(list(zip(category_ids, category_names)), columns=['category_id', 'name'])
    metadata = pd.merge(metadata, category_df, how='inner', left_on=['category_id'], right_on=['category_id']) # [location', 'date', 'image_id', 'category_id', 'filename', 'name']
    
    print('len(metadata) = {}'.format(len(metadata)))

    print(metadata.columns)

    if args.split_dataset:
        splits = ['train'] * len(metadata)
        print('len(splits) = {}'.format(len(splits)))

        n_val_samples = int(0.15 * len(metadata))
        n_test_samples = int(0.15 * len(metadata))
        splits[:n_val_samples] = ['val']*n_val_samples
        splits[n_val_samples : n_val_samples+n_test_samples] = ['test']*n_test_samples
        random.shuffle(splits)

        print('len(splits) = {}'.format(len(splits)))
        print(splits.count('train'))
        print(splits.count('val'))
        print(splits.count('test'))

        metadata = metadata.assign(split=splits)

    # create category_id_to_name
    category_id_to_name = {x['id']:x['name'] for x in category}

    taxon = pd.read_csv("snapshot_mountain_zebra/taxon.csv")
    print('len(taxon) = {}'.format(len(taxon)))

    if not args.no_location:
        takeLocation = metadata.loc[:, ['filename', 'location', 'split']]
        takeLocation.insert(loc=1, column='r', value=2)
        takeLocation.insert(loc=1, column='datatype_h', value='image')
        takeLocation.insert(loc=4, column='datatype_t', value='location')
        takeLocation.insert(loc=6, column='dataset', value=args.dataset_prefix)
        takeLocation.columns = ['h', 'datatype_h', 'r', 't', 'datatype_t', 'split', 'dataset']
        print(takeLocation.head())

    if not args.no_datetime:
        takeTime = metadata.loc[:, ['filename', 'datetime', 'split']]
        takeTime.insert(loc=1, column='r', value=0)
        takeTime.insert(loc=1, column='datatype_h', value='image')
        takeTime.insert(loc=4, column='datatype_t', value='time')
        takeTime.insert(loc=6, column='dataset', value=args.dataset_prefix)
        takeTime.columns = ['h', 'datatype_h', 'r', 't', 'datatype_t', 'split', 'dataset']
        print(takeTime.head())

    imageIsIn = metadata.loc[:, ['filename', 'name', 'split']]
    imageIsIn.insert(loc=1, column='r', value=3)
    imageIsIn.insert(loc=1, column='datatype_h', value='image')
    imageIsIn.insert(loc=4, column='datatype_t', value='id')
    imageIsIn.insert(loc=6, column='dataset', value=args.dataset_prefix)
    imageIsIn.columns = ['h', 'datatype_h', 'r', 't', 'datatype_t', 'split', 'dataset']
    print(imageIsIn.head())

    dataset = pd.concat([taxon, imageIsIn, takeTime, takeLocation], ignore_index=True)
    
    out_file = os.path.join(args.data_dir, 'data_triples.csv')
    dataset.to_csv(out_file, index=False)

