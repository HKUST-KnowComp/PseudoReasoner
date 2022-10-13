"""
This file is different from train_gpt2.py from that:
    1. The input should not be the .csv files from CSKB Population (with label 0 and 1). 
       The all the triples in the input.csv will be regarded as training examples.
    2. Save model by training steps, not the best AUC.
"""
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
from models.model_utils import evaluate
from models.dataloader import CKBPDataset

# GPT-2
# CUDA_VISIBLE_DEVICES=0 PRED_DATA_PATH=data/ckbp_csv/emnlp2021/train_pos_unique_1000.csv OUT_DIR=results TOKENIZER=results/gpt2_bs64/best_tokenizer GPT2_MODEL=results/gpt2_bs64 NUM_GEN=1 python models/gpt2/generate_comet.py

def main():

    config = argparse.Namespace()

    config.IN_LENGTH = int(os.environ.get("IN_LENGTH", 16))
    config.OUT_LENGTH = int(os.environ.get("IN_LENGTH", 34))

    config.SEED = int(os.environ.get("SEED", 100))

    config.OUT_DIR = os.environ.get("OUT_DIR", "results")

    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl") # "data/models/gpt2xl-comet-atomic-2020/tokenizer/"
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-xl")
    config.USE_NL_RELATION = os.environ.get('USE_NL_RELATION', "False")=="True" # whether to use natural language descriptions for the relations.
    config.NUM_GEN = int(os.environ.get("NUM_GEN", 10))
    config.PRED_DATA_PATH = os.environ.get('PRED_DATA_PATH', "data/ckbp_csv/emnlp2021/eval.csv")

    config.device = "cuda"

    ## logging

    save_dir = config.OUT_DIR

    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger()


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

    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'pad_token': '[PAD]',
        'additional_special_tokens': special_token_list
    })

    pred_dataset = pd.read_csv(config.PRED_DATA_PATH)

    pred_dataset = pred_dataset.rename(columns={"head_event": "head", "tail_event": "tail"})

    pred_dataset = pred_dataset.drop_duplicates(subset=["head", "relation"])

    if config.USE_NL_RELATION:
        pred_dataset["relation"] = pd.Series(map(lambda r: CS_RELATIONS_2NL[r], pred_dataset["relation"]))

    pred_dataset = CKBPDataset(pred_dataset, tokenizer, config.IN_LENGTH, model="comet_gpt2", is_eval=True)

    pred_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    }

    pred_dataloader = DataLoader(pred_dataset, **pred_params, drop_last=True)

    logger.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name)
    logger.info("Move model to device {}".format(config.device))
    model = model.to(config.device)
    model.resize_token_embeddings(len(tokenizer))


    model.eval()
    predictions = []
    actuals = []
    sources = []
    pred_generations = []
    

    with torch.no_grad():
        for _, data in tqdm(enumerate(pred_dataloader, 0)):
            ids = data['ids'].to(config.device, dtype=torch.long)
            mask = data['mask'].to(config.device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=1.0,
                do_sample=False,
                max_length=config.OUT_LENGTH,
                top_p=0.9,
                top_k=40, # default
                repetition_penalty=1.0,
                # num_return_sequences=10 if top_k > 1 else 1,
                num_return_sequences=config.NUM_GEN,
                num_beams=10
            )

            preds = [tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in generated_ids]
            try:
                target = [tokenizer.decode(t, clean_up_tokenization_spaces=True) for t in y]
            except:
                target = ['']
            source = [tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in ids]

            pred_generations.append({
                'source': source[0],
                'target': target[0],
                'generations': preds
            })

            if _ % 100 == 0:
                logger.info(f'Completed {_}')
    def write_items(output_file, items):
        with open(output_file, 'w') as f:
            for concept in items:
                f.write(concept + "\n")
        f.close()

    write_items(os.path.join(config.OUT_DIR, 
        os.path.basename(model_name)+"_"+os.path.basename(config.PRED_DATA_PATH).split(".")[0]+"_pred_generations.jsonl"),
        [json.dumps(r) for r in pred_generations])
        

if __name__ == '__main__':

    main()
