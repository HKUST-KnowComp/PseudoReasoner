import sys, os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import argparse

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from utils.ckbp_utils import CS_RELATIONS_2NL, atomic18_relations, num_of_triples_emnlp21
from models.dataloader import CKBPDataset
from models.model_utils import score_triples


if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'

    config = argparse.Namespace()
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 32))
    config.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 34))

    config.USE_NL_RELATION = os.environ.get('USE_NL_RELATION', "True")=="True"
    config.RELATION = os.environ.get('RELATION', "all") 
    # relation in consideration, can be atomic18, or a sequence of relation xAttr-xIntent-xNeed ('-' separated for mulitple relation)
    
    config.NUM_SAMPLES_DEBUG = int(os.environ.get("NUM_SAMPLES_DEBUG", -1))
    config.PROCESS_MAX_SAMPLES = int(os.environ.get("PROCESS_MAX_SAMPLES", 2e6))
    # to avoid processing too much data while it's unnecessary

    config.DATA_WITH_MODEL2_LOSS_DIR = os.environ.get('DATA_WITH_MODEL2_LOSS_DIR', "./data/DISCOS_inference/gpt2_large_nl/")
    config.OUTPUT_DIR = os.environ.get('OUTPUT_DIR', "./data/CSKB_candidates_scores/gpt2_xl_nl")

    config.MODEL1_NAME = os.environ.get('MODEL1_NAME', "gpt2-xl")

    ## checking i/o
    if not os.path.exists(config.DATA_WITH_MODEL2_LOSS_DIR):
        print('ERROR: Cannot find data with commonsense loss at {}'.format(config.DATA_WITH_MODEL2_LOSS_DIR))
        exit(-1)
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    
    ## load model
    model1_name = config.MODEL1_NAME
    print('Loading model 1 {} ...'.format(model1_name))
    model1 = GPT2LMHeadModel.from_pretrained(model1_name).to(device)
    print('Loading completed')

    tokenizer1 = GPT2Tokenizer.from_pretrained(model1_name)
    tokenizer1.add_special_tokens({
        'eos_token': '[EOS]',
        'pad_token': '[PAD]'
    })
    model1.resize_token_embeddings(len(tokenizer1))

    ## configure the dataset
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    if config.RELATION == "atomic18":
        rel_list = atomic18_relations
    elif config.RELATION == "all":
        rel_list = list(num_of_triples_emnlp21.keys())
    else:
        rel_list = config.RELATION.split('-')

    for r in rel_list:
        print('Processing data of relation {}'.format(r))

        ## load data (ASER candidates of relation r)
        infer_file = pd.read_csv(os.path.join(config.DATA_WITH_MODEL2_LOSS_DIR, r + '.csv')) # head, tail, score
        infer_file = infer_file.rename(columns={
            "score": "commonsense_loss"
        })
        # infer_file = infer_file.drop(["score"], axis=1)
        
        if config.PROCESS_MAX_SAMPLES > 0:
            infer_file = infer_file.head(config.PROCESS_MAX_SAMPLES)
        if config.NUM_SAMPLES_DEBUG > 0:
            infer_file = infer_file.head(config.NUM_SAMPLES_DEBUG)
        print(f'Processing {len(infer_file)} triples')
        
        infer_file.insert(1, "relation", CS_RELATIONS_2NL.get(r,r) if config.USE_NL_RELATION else r)
        print(infer_file.head(10))

        candidate_set = CKBPDataset(infer_file, tokenizer1, config.MAX_LENGTH, model="gpt2") 
        candidate_loader = DataLoader(candidate_set, **val_params, drop_last=False)

        ## calc scores
        model1_negloss_list = score_triples(tokenizer1, model1, device, candidate_loader, model_type="gpt2")

        ## add loss to the data frame and save
        infer_file.insert(len(infer_file.columns), "fluency_loss", model1_negloss_list)
        infer_file["fluency_loss"] = -infer_file["fluency_loss"]
        infer_file = infer_file.drop(["relation"], axis=1)
        infer_file.to_csv(os.path.join(config.OUTPUT_DIR, r+".csv"), index=False)