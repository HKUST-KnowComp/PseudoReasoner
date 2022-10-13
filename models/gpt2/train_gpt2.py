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
# CKBP: GPT2-S: CUDA_VISIBLE_DEVICES=2 TRAIN_DATA_PATH=data/for_gpt/train.csv FILE_TYPE=csv EVAL_TASK=Population EVAL_EVERY=250 USE_NL_RELATION=False TRAIN_BATCH_SIZE=32 VALID_BATCH_SIZE=32 DO_TRAIN=True DO_PRED=False OUT_DIR=data/models/comet/gpt2 TOKENIZER=gpt2 GPT2_MODEL=gpt2 EVAL_METRIC=overall_auc SAVE_BEST=False python model/comet/comet_gpt2.py
# CKBP: CUDA_VISIBLE_DEVICES=1 TOKENIZER=gpt2 GPT2_MODEL=gpt2 EVAL_EVERY=250 python models/gpt2/train_gpt2.py

def main():

    config = argparse.Namespace()
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 32)) 
    config.GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1)) 
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 128)) 
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 1))
    config.MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", -1))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))

    config.MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 30))

    config.SEED = int(os.environ.get("SEED", 100))

    config.OUT_DIR = os.environ.get("OUT_DIR", "results")

    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl") # "data/models/gpt2xl-comet-atomic-2020/tokenizer/"
    config.MODEL_NAME = os.environ.get('GPT2_MODEL', "gpt2-xl")
    config.EVAL_EVERY = int(os.environ.get('EVAL_EVERY', 250))
    config.USE_NL_RELATION = os.environ.get('USE_NL_RELATION', "False")=="True" # whether to use natural language descriptions for the relations.
    config.POPULATION_EVAL_FILE = os.environ.get('POPULATION_EVAL_FILE', "data/evaluation_set.csv") 

    config.EVAL_METRIC = os.environ.get("EVAL_METRIC", "overall_auc")
    config.SAVE_BEST = os.environ.get("SAVE_BEST", "False") == "True"
    config.RELATION_AS_SPECIAL_TOKEN = os.environ.get("RELATION_AS_SPECIAL_TOKEN", "True") == "True" # default is true.

    config.device = "cuda"

    ## logging
    model_name_strip = os.path.basename(config.MODEL_NAME)

    save_dir = os.path.join(config.OUT_DIR, "_".join([model_name_strip, f"bs{config.TRAIN_BATCH_SIZE * config.GRAD_ACCUM_STEPS}", 
                            f"evalstep{config.EVAL_EVERY}", "rel_special" if config.RELATION_AS_SPECIAL_TOKEN else ""]))

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

    train_dataset = pd.read_csv(
        os.environ.get('TRAIN_DATA_PATH', "data/ckbp_csv/emnlp2021/train.csv")
    )

    train_dataset = train_dataset.rename(columns={"head_event": "head", "tail_event": "tail"})
    # in GPT2, only positive examples are selected.
    train_dataset = train_dataset[train_dataset["label"]==1]


    if config.USE_NL_RELATION:
        train_dataset["relation"] = pd.Series(map(lambda r: CS_RELATIONS_2NL[r], train_dataset["relation"]))

    training_set = CKBPDataset(train_dataset, tokenizer, config.MAX_LENGTH, model="gpt2")

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
    infer_file = pd.read_csv(config.POPULATION_EVAL_FILE)
    
    dev_dataset = CKBPDataset(infer_file[infer_file["split"] == "dev"], tokenizer, config.MAX_LENGTH, model="gpt2") 
    tst_dataset = CKBPDataset(infer_file[infer_file["split"] == "tst"], tokenizer, config.MAX_LENGTH, model="gpt2") 

    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)
    tst_dataloader = DataLoader(tst_dataset, **val_params, drop_last=False)

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
                del loss, ids, mask

            # eval:

            if iteration > 0 and (iteration // config.GRAD_ACCUM_STEPS) % config.EVAL_EVERY == 0\
                 or (iteration // config.GRAD_ACCUM_STEPS) in [1,3,10,30, 50, 75, 100, 125, 150, 175, 200]:
                # 
                model.eval()

                eval_auc, _ = evaluate(tokenizer, model, config.device, dev_dataloader, model_type="gpt2")

                assert _ == len(dev_dataset)

                if eval_auc > best_val_score:
                    best_val_score = eval_auc
                    if config.SAVE_BEST:
                        model.save_pretrained(save_dir + f"/best_model_seed_{config.SEED}")
                        tokenizer.save_pretrained(save_dir + "/best_tokenizer")
                    
                    best_epoch, best_iter = e, iteration

                    # calc test scores

                    tst_auc, _, class_scores, _ = evaluate(tokenizer, model, config.device, tst_dataloader, class_break_down=True, model_type="gpt2")

                    logger.info(f"Overall auc & Test Set & CSKB Head + ASER tail & ASER edges. Reached at epoch {best_epoch} step {best_iter}")
                    logger.info("test scores:" + " & ".join([str(round(tst_auc*100, 1))]+\
                            [str(round(class_scores[clss]*100, 1)) for clss in ["test_set", "cs_head", "all_head"]]) )

                model.train()
        if iteration > config.MAX_TRAIN_STEPS and config.MAX_TRAIN_STEPS > 0:
            break

if __name__ == '__main__':

    main()
