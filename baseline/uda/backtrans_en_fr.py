import os, sys
sys.path.append(os.getcwd())

import math
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader

from utils.ckbp_utils import CS_RELATIONS_2NL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsup_data_dir", 
                        default="data/pseudo_label_trn", 
                        type=str, required=True)
    parser.add_argument("--use_nl_relation", action="store_true")
    parser.add_argument("--translate_to", 
                        default='fr', 
                        type=str, required=False,
                        help='the language that we translate '
                        'English sample to, then translate back to English')
    parser.add_argument("--batch_size", 
                        default=32, 
                        type=int, required=False)
    parser.add_argument("--sep_token", 
                        default=" ", 
                        type=str, required=False)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    personx_to_realname = [
        ['PersonX', 'Alice'],['PersonY', 'Charles'],['PersonZ', 'Francisco']]
    
    ## load data
    print('loading data ...')
    df = pd.read_csv(args.unsup_data_dir)
    if args.use_nl_relation:
        df['relation'] = df['relation'].apply(lambda x: CS_RELATIONS_2NL.get(x, x))
    examples = df['head']+args.sep_token+df['relation']+args.sep_token+df['tail']
    for px, rn in personx_to_realname:
        examples = examples.apply(lambda x: x.replace(px, rn))
    examples = list(examples)
    if args.debug:
        examples = examples[:100]

    ## load models
    print('loading models ...')
    fw_translator = pipeline("translation", 
                            model=f"Helsinki-NLP/opus-mt-en-{args.translate_to}",
                            device=0)
    bw_translator = pipeline("translation", 
                            model=f"Helsinki-NLP/opus-mt-{args.translate_to}-en",
                            device=0)

    ## perform forward and backward translation
    # using Dataset+DataLoader is actually slower than naive implementation
    # best: 2 samples/sec -> have to take a small set of unsup 3x
    print('start translation ...')
    augment_examples = []
    num_of_iter = math.ceil(len(examples)/args.batch_size)
    for i in tqdm(range(num_of_iter), total=num_of_iter):
        tmp = examples[i*args.batch_size:(i+1)*args.batch_size]
        tmp = fw_translator(tmp)
        tmp = [x['translation_text'] for x in tmp]
        tmp = bw_translator(tmp)
        augment_examples.extend([x['translation_text'] for x in tmp])

    ## postprocess, save augment data, add 'tail' and 'relation' to avoid error
    print('postprocess data ...')
    output = pd.DataFrame({'head': augment_examples})
    for px, rn in personx_to_realname:
        output['head'] = output['head'].apply(lambda x: x.replace(rn, px))
    output['tail'] = ''
    output['relation'] = ''
    output.to_csv(args.unsup_data_dir[:-4]+\
        f"{'_NLrel' if args.use_nl_relation else ''}_bt_en2{args.translate_to}.csv", 
        index=False)
    print('done')


if __name__ == "__main__":
    main()