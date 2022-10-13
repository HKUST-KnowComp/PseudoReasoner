import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


import logging
logging.basicConfig(level=logging.INFO)

from utils.ckbp_utils import CS_RELATIONS_2NL, all_relations, special_token_list
from models.model_utils import score_triples
from models.dataloader import CKBPDataset

# --- overall AUC
# CKBP: 
# CUDA_VISIBLE_DEVICES=0 TOKENIZER=results/gpt2_bs32_evalstep250_rel_special/best_tokenizer GPT2_MODEL=results/gpt2_bs32_evalstep250_rel_special/best_model_seed_100 CANDIDATE_SCORING_FILE=data/discos_csv/candidates/test.csv SCORER_NAME=gpt2_bs32 python models/gpt2/scoring_gpt2.py

def main():

    config = argparse.Namespace()
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 128)) 
    config.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 30))

    config.SEED = int(os.environ.get("SEED", 100))

    config.OUT_DIR = os.environ.get("OUT_DIR", "results/scores")

    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl") 
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-xl")
    config.SCORER_NAME = os.environ.get('SCORER_NAME', "UNK") # name of the LM. self-defined.
    config.USE_NL_RELATION = os.environ.get('USE_NL_RELATION', "False")=="True" # whether to use natural language descriptions for the relations.
    config.RELATION_AS_SPECIAL_TOKEN = os.environ.get("RELATION_AS_SPECIAL_TOKEN", "True") == "True" # default is true.

    config.CANDIDATE_SCORING_FILE = os.environ.get('CANDIDATE_SCORING_FILE', "")
    config.device = "cuda"

    ## logging

    save_dir = os.path.join(config.OUT_DIR, config.SCORER_NAME) 
    # For fine-tuned GPT2, use RELATION_AS_SPECIAL_TOKEN == True
    # For original GPT2, use USE_NL_RELATION == True.
    # Or accordingly other settings.

    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
    logger.addHandler(file_handler)

    ## set random seeds:
    random.seed(config.SEED)
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True

    ## model

    model_name = config.MODEL_NAME

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER)

    if config.RELATION_AS_SPECIAL_TOKEN:
        tokenizer.add_special_tokens({
            'eos_token': '[EOS]',
            'additional_special_tokens': special_token_list
        })
    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'pad_token': '[PAD]'
    })


    # loadin eval dataset.    
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 5
    }
    infer_file = pd.read_csv(config.CANDIDATE_SCORING_FILE)

    if config.USE_NL_RELATION:
        logger.info("using natural language as relations.")
        infer_file["relation"] = infer_file["relation"].apply(lambda r:CS_RELATIONS_2NL[r])
    
    eval_dataset = CKBPDataset(infer_file, tokenizer, config.MAX_LENGTH, model="gpt2") 

    eval_dataloader = DataLoader(eval_dataset, **val_params, drop_last=False)

    logging.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name)
    logging.info("Move model to device {}".format(config.device))
    model = model.to(config.device)
    model.resize_token_embeddings(len(tokenizer))

    predicted_scores = score_triples(tokenizer, model, config.device, eval_dataloader, model_type="gpt2")

    # write to file:

    infer_file["score"] = predicted_scores

    infer_file.to_csv(os.path.join(save_dir, os.path.basename(config.CANDIDATE_SCORING_FILE)), index=False)

if __name__ == '__main__':

    main()
