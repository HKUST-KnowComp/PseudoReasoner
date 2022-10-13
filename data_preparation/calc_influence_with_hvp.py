import os
import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import RandomSampler

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

sys.path.append(os.getcwd())

from models.pseudo_labeling.model import KGBERTClassifier
from models.model_utils import score_triples
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer

from utils.ckbp_utils import special_token_list

def get_influence(args, eval_dataloader, model, HVP):

    # Eval!

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    random_state = torch.get_rng_state()
    model.zero_grad()

    HVP = [it.cuda() for it in HVP]
    no_decay = ['bias', 'LayerNorm.weight']
    count = 0
    negative_count = 0
    influence_list = []
    for data in tqdm(eval_dataloader, desc="Calculating validation grad", position=0):
        model.eval()

        y = data['label'].to(args.device, dtype=torch.long)

        ids = data['ids'].to(args.device, dtype=torch.long)
        mask = data['mask'].to(args.device, dtype=torch.long)
        tokens = {"input_ids":ids, "attention_mask":mask}
        outputs = model(tokens)

        logits = outputs

        loss = F.cross_entropy(logits, y, reduction='mean')
        loss.backward()
        count += 1
        influence = 0
        for i, ((n, p), v) in enumerate(zip([(name, p) for name, p in model.named_parameters() if not name.startswith("model.pooler.dense")], 
                                            HVP)):
            if p.grad is None:
                print("wrong")
            else:
                if not any(nd in n for nd in no_decay):
                    influence += (
                        (p.grad.data.add_(args.weight_decay, p.data)) *
                        v).sum() * -1


    #                    influence += ((p.grad.data)*v).sum() * -1
                else:
                    influence += ((p.grad.data) * v).sum() * -1

        if influence.item() < 0:
            negative_count += 1
        influence_list.append(influence.item())
        if count % 100 == 0:
            # print(influence.item())
            print(negative_count / count)
    influence_list = np.array(influence_list)
    return influence_list

def parse_args():
    parser = argparse.ArgumentParser()


    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str, 
                        required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--checkpoint_path", required=False, default="",
                    help="loading a checkpoint") # a bit different from --ptlm.
    group_model.add_argument("--tokenizer_path", required=False, default="",
                    help="loading tokenizer")
    
    group_model.add_argument("--candidate_file_path", required=False, default="",
                    help="candidate csv")
    group_model.add_argument("--device", required=False, default="cuda")
    group_model.add_argument("--weight_decay", required=False, default=0.0, type=float,
                    help="loading tokenizer")
    group_model.add_argument("--hvp_path", required=False, default="",
                    help="hvp_path")


    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_model.add_argument("--start", required=True, type=int)
    group_model.add_argument("--end", required=True, type=int)

    
    group_data.add_argument("--seed", default=100, type=int, required=False,
                    help="random seed")

    args = parser.parse_args()

    return args


def main():


    args = parse_args()


    model = KGBERTClassifier(args.ptlm).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(args.checkpoint_path))

    sep_token = tokenizer.sep_token

    HVP = torch.load(args.hvp_path)

    candi_dataset = pd.read_csv(args.candidate_file_path).loc[args.start:args.end-1] # .loc[a:b] is [a:b+1]
    candi_params = {
            'batch_size': 1,
            'num_workers': 1
    }
    max_length=30
    candi_dataset = CKBPDataset(candi_dataset, tokenizer, max_length, sep_token=sep_token)

    candi_loader = DataLoader(candi_dataset, **candi_params, drop_last=True)

    print('Start getting lissa influence score ...')
    influence_list = get_influence(args, candi_loader, model, HVP)

    hvp_dir = os.path.dirname(args.hvp_path)
    hvp_name = os.path.basename(args.hvp_path)[:-4] # seed_{args.seed}
    out_dir = os.path.join(hvp_dir, f"influence_weight_decay_{args.weight_decay}", 
                    os.path.basename(args.candidate_file_path), hvp_name )

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"start_{args.start}_end_{args.end}"), influence_list)

if __name__ == "__main__":
    main()
