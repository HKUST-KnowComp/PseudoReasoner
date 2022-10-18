"""
Input: a set of unlabel data
Output: a replica of the dataset after if-idf replacement
"""

import json
import os
import argparse
import random

import numpy as np
import pandas as pd

import word_level_augment
from tokenization import BasicTokenizer


def build_vocab(examples):
    all_words = []
    for eg in examples:
        all_words.extend(eg)
    all_words = sorted(list(set(all_words)))

    vocab = {}
    for i, w in enumerate(all_words):
        vocab[i] = w
    return vocab

def get_data_stats(data_stats_dir, examples):
    keys = ["tf_idf", "idf"]
    all_exist = True

    for key in keys:
        data_stats_path = "{}/{}.json".format(data_stats_dir, key)
        if not os.path.exists(data_stats_path):
            all_exist = False
        print("Not exist: {}".format(data_stats_path))
    if all_exist:
        print("loading data stats from {:s}".format(data_stats_dir))
        data_stats = {}
        for key in keys:
            with open(
                "{}/{}.json".format(data_stats_dir, key)) as inf:
                data_stats[key] = json.load(inf)
    else:
        data_stats = word_level_augment.get_data_stats(examples)
        os.makedirs(data_stats_dir)
        for key in keys:
            with open("{}/{}.json".format(data_stats_dir, key), "w") as ouf:
                json.dump(data_stats[key], ouf)
        print("dumped data stats to {:s}".format(data_stats_dir))
    return data_stats

def parse_args():
    parser = argparse.ArgumentParser()

    # unsup_data_dir, default 
    # "data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/"+
    # "hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv"
    parser.add_argument("--unsup_data_dir", 
                        default="data/pseudo_label_trn", 
                        type=str, required=True)
    parser.add_argument("--seed", default=101, type=int, required=False,
                        help="random seed")
    parser.add_argument("--aug_ops", 
                        default="tf_idf-0.1", 
                        type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--sep_token", 
                        default=" ", 
                        type=str, required=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    ## load unsup data and perform basic word tokenization
    df = pd.read_csv(args.unsup_data_dir)
    examples = df['head']+args.sep_token+df['relation']+args.sep_token+df['tail']

    tokenizer = BasicTokenizer(do_lower_case=args.do_lower_case)
    for i in range(len(examples)):
        examples[i] = tokenizer.tokenize(examples[i])
        # string -> list of tokens
    print("finished tokenizing examples")
    
    ## get data stats
    data_stats_dir = os.path.join(os.path.dirname(args.unsup_data_dir), 
                                  os.path.basename(args.unsup_data_dir)[:-4]+"_data_stats")

    data_stats = get_data_stats(data_stats_dir, examples)
    

    ## run augmentation
    print("building vocab list of unsup data")
    word_vocab = build_vocab(examples)
    examples = word_level_augment.word_level_augment(
        examples, args.aug_ops, word_vocab, data_stats
    )
    examples = examples.apply(lambda x: " ".join(x))

    ## save augment data, add 'tail' and 'relation' to avoid error
    output = pd.DataFrame({'head': examples})
    output['tail'] = ''
    output['relation'] = ''
    output.to_csv(args.unsup_data_dir[:-4]+'_'+args.aug_ops+'.csv', index=False)
    

if __name__ == "__main__":
    main()
