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

# --- overall AUC
# CUDA_VISIBLE_DEVICES=1 TRAIN_BATCH_SIZE=64 TRAIN_EPOCHS=1 TOKENIZER=gpt2 GPT2_MODEL=gpt2 python models/gpt2/train_comet_gpt2.py

def main():

    config = argparse.Namespace()
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32)) 
    config.GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1)) 
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 128)) 
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 1)) # 3
    config.MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", -1))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))

    config.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 30))

    config.SEED = int(os.environ.get("SEED", 100))

    config.OUT_DIR = os.environ.get("OUT_DIR", "results")

    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl") # "data/models/gpt2xl-comet-atomic-2020/tokenizer/"
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-xl")
    config.USE_NL_RELATION = os.environ.get('USE_NL_RELATION', "False")=="True" # whether to use natural language descriptions for the relations.

    config.SAVE_LAST_CHECKPOINT = os.environ.get("SAVE_LAST_CHECKPOINT", "True") == "True"

    config.device = "cuda"

    ## logging
    model_name_strip = os.path.basename(config.MODEL_NAME)

    save_dir = os.path.join(config.OUT_DIR, "_".join([model_name_strip, f"bs{config.TRAIN_BATCH_SIZE * config.GRAD_ACCUM_STEPS}"]))

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

    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'pad_token': '[PAD]',
        'additional_special_tokens': special_token_list
    })

    train_dataset = pd.read_csv(
        os.environ.get('TRAIN_DATA_PATH', "data/ckbp_csv/emnlp2021/train_pos.csv")
    )

    train_dataset = train_dataset.rename(columns={"head_event": "head", "tail_event": "tail"})


    if config.USE_NL_RELATION:
        train_dataset["relation"] = pd.Series(map(lambda r: CS_RELATIONS_2NL[r], train_dataset["relation"]))

    training_set = CKBPDataset(train_dataset, tokenizer, config.MAX_LENGTH, model="comet_gpt2")

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 5
    }

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    
    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 5
    }

    logger.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name)
    logger.info("Move model to device {}".format(config.device))
    model = model.to(config.device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    optimizer.zero_grad()

    best_val_score = 0

    iteration = 0

    for e in range(config.TRAIN_EPOCHS):

        for iteration, data in tqdm(enumerate(training_loader, iteration+1)):

            if iteration > config.MAX_TRAIN_STEPS and config.MAX_TRAIN_STEPS > 0:
                logger.info(f"stop training at step {config.MAX_TRAIN_STEPS}")
                break

            ids = data['ids'].to(config.device, dtype=torch.long)
            mask = data['mask'].to(config.device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
            
            loss = outputs[0]

            
            loss.backward()
            if iteration % config.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        if iteration > config.MAX_TRAIN_STEPS and config.MAX_TRAIN_STEPS > 0:
            break
    if config.SAVE_LAST_CHECKPOINT:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir + "/best_tokenizer")

if __name__ == '__main__':

    main()
