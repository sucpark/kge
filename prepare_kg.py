import os
import json
import pickle
import argparse
from pathlib import Path
import torchkge.utils.datasets as torchkge_ds

parser = argparse.ArgumentParser(description='Training knowledge graph using development knowledge base')
parser.add_argument('--data_dir', default='raw_data', help='Directory containing raw data')
parser.add_argument('--save_dir', default='data', help='Directory to save the refined kg')
parser.add_argument('--data', default='wikidatasets')
parser.add_argument('--share', default=0.8, help='The proportion of training triplets')

parser_for_kg_wiki = parser.add_argument_group(title='wiki')
parser_for_kg_wiki.add_argument('--which', default='companies')
parser_for_kg_wiki.add_argument('--limit', default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) / args.data

    assert args.data in ['wikidatasets', 'fb15k'], "Invalid knowledge graph dataset"
    if args.data == 'wikidatasets':
        kg = torchkge_ds.load_wikidatasets(which=args.which,
                                           limit_=args.limit,
                                           data_home=args.data_dir)
        save_dir = save_dir / args.which
    elif args.data == 'fb15k':
        kg = torchkge_ds.load_fb15k(data_home=args.data_dir)

    kg_train, kg_valid, kg_test = kg.split_kg(share=args.share, validation=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(save_dir / 'kg_train.pkl', mode='wb') as io:
        pickle.dump(kg_train, io)
    with open(save_dir / 'kg_valid.pkl', mode='wb') as io:
        pickle.dump(kg_valid, io)
    with open(save_dir / 'kg_test.pkl', mode='wb') as io:
        pickle.dump(kg_test, io)
    with open(save_dir / 'summary_data.json', mode='w') as io:
        json.dump(
            dict(
                triplet_count=kg.n_facts,
                entity_count=kg.n_ent,
                relation_count=kg.n_rel,
                train_triplet_count=len(kg_train.head_idx),
                validation_triplet_count=len(kg_valid.head_idx),
                test_triplet_count=len(kg_test.head_idx)
            ),
            io,
            indent=4,
        )




