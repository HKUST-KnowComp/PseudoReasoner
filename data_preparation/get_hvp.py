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

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

sys.path.append(os.getcwd())

from models.pseudo_labeling.model import KGBERTClassifier
from models.model_utils import score_triples
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer

from utils.ckbp_utils import special_token_list

def get_validation_grad(model, eval_dataloader, device="cuda"):

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    model.eval()
    model.zero_grad()
    
    #eval_dataloader = eval_dataloader[:10]

    for data in tqdm(eval_dataloader, desc="Calculating validation grad"):
        y = data['label'].to(device, dtype=torch.long)
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        
        tokens = {"input_ids":ids, "attention_mask":mask}
        
        
        
        outputs = model(tokens)
        logits = outputs
        loss = F.cross_entropy(logits, y, reduction='sum')
        loss.backward()

    grad = []
    for name, p in model.named_parameters():
        if not name.startswith("model.pooler.dense"):
            grad.append((p.grad.data / len(eval_dataloader)).cpu())
        else:
            print("not used params:", name)

    return grad


from tqdm import trange

# train_dataloader = training_loader
# model = model
# v = grad

def get_HVP(args, train_dataloader, model, v):


    no_decay = ['bias', 'LayerNorm.weight']
    final_res = None
    damping = args.damping # 0
    gradient_accumulation_steps = args.gradient_accumulation_steps # 10
    weight_decay = args.weight_decay # 0.01
    c = args.c # 1e7
    epoch = args.epoch

    for r in trange(epoch): # epoch. args.r
        res = [w.clone().cuda() for w in v]
        model.zero_grad()
        for step, data in enumerate(
                tqdm(train_dataloader, desc="Calculating HVP"), position=0):
            model.eval()

            y = data['label'].to(args.device, dtype=torch.long)

            ids = data['ids'].to(args.device, dtype=torch.long)
            mask = data['mask'].to(args.device, dtype=torch.long)
            tokens = {"input_ids":ids, "attention_mask":mask}
            outputs = model(tokens)

            logits = outputs

            loss = F.cross_entropy(logits, y, reduction='mean')

            grad_list = torch.autograd.grad(loss,
                                            [p for name, p in model.named_parameters() if not name.startswith("model.pooler.dense")],
                                            create_graph=True)

            grad = []

            H = 0
            for i, (g, g_v) in enumerate(zip(grad_list, res)):

                H += (g * g_v).sum() / gradient_accumulation_steps
            #H = grad @ v
            H.backward()

            #grad = []
            if (step + 1) % gradient_accumulation_steps == 0:

                for i, ((n, p),
                        v_p) in enumerate(zip([(name, p) for name, p in model.named_parameters() if not name.startswith("model.pooler.dense")], res)):
                    try:
                        if not any(nd in n for nd in no_decay):
                            res[i] = (1 - damping) * v_p - (
                                p.grad.data.add_(weight_decay,
                                                 v_p)) / c + v[i].cuda()
                        else:
                            res[i] = (1 - damping) * v_p - (
                                p.grad.data) / c + v[i].cuda()
                    except RuntimeError:

                        v_p = v_p.cpu()

                        p_grad = p.grad.data.cpu()

                        if not any(nd in n for nd in no_decay):
                            res[i] = ((1 - damping) * v_p -
                                      (p_grad.add_(weight_decay, v_p)) /
                                      c + v[i]).cuda()
                        else:
                            res[i] = ((1 - damping) * v_p -
                                      (p_grad) / c + v[i]).cuda()
                model.zero_grad()
#             if step > 20:
#                 break

        if final_res is None:
            final_res = [(b / c).cpu().float() for b in res]
        else:
            final_res = [
                a + (b / c).cpu().float() for a, b in zip(final_res, res)
            ]

    final_res = [a / float(epoch) for a in final_res]
    return final_res

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
    
    group_model.add_argument("--train_file_path", required=False, default="",
                    help="training csv")
    group_model.add_argument("--eval_file_path", required=False, default="",
                    help="evaluation csv")

    group_model.add_argument("--device", required=False, default="cuda")
    
    

    # hyperparameters for HVP
    group_model.add_argument("--num_train", default=10000, type=int, required=False,
                    help="number of training examples used")
    group_model.add_argument("--weight_decay", required=False, default=0.01, type=float,
                    help="loading tokenizer")
    group_model.add_argument("--damping", required=False, default=0.01, type=float,
                    help="loading tokenizer")
    group_model.add_argument("--gradient_accumulation_steps", default=10, type=int, required=False,
                    help="grad accum steps")
    group_model.add_argument("--c", default=1e7, type=float, required=False,
                    help="scale")
    group_model.add_argument("--epoch", default=10, type=int, required=False,
                    help="epoch of the hvp approximation")

    # IO-related

    group_data = parser.add_argument_group("IO related configs")


    
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

    val_params = {
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 5
    }
    infer_file = pd.read_csv(args.eval_file_path)
    max_length=30
    dev_dataset = CKBPDataset(infer_file[infer_file["split"] == "dev"], tokenizer, max_length, sep_token=sep_token) 

    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)

    # training
    train_dataset = pd.read_csv(args.train_file_path)
    train_params = {
            'batch_size': 16,
            'num_workers': 1
    }

    training_set = CKBPDataset(train_dataset, tokenizer, max_length, sep_token=sep_token)
    training_sampler = RandomSampler(training_set,
                                      replacement=True,
                                      num_samples=args.num_train)

    training_loader = DataLoader(training_set, **train_params, sampler=training_sampler, drop_last=True)

    grad = get_validation_grad(model, dev_dataloader)

    HVP = get_HVP(args, training_loader, model, grad)

    torch.save(HVP,
        os.path.join(os.path.dirname(args.checkpoint_path),
            f"hvp_{args.num_train}_damping_{args.damping}_weight_decay_{args.weight_decay}_c_{args.c}_epoch_{args.epoch}.pth", 
        )
    )


if __name__ == "__main__":
    main()