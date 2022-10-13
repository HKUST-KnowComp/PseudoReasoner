import os
import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

sys.path.append(os.getcwd())

from models.pseudo_labeling.model import KGBERTClassifier
from models.model_utils import score_triples
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO)

from utils.ckbp_utils import special_token_list

def get_validation_grad(model, eval_dataloader, device="cuda"):
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    
    model.eval()
    model.zero_grad()
    
    # eval_dataloader = eval_dataloader[:10]
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


################ functions for influence function ################
def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def hv(loss, model_params, v): # according to pytorch issue #24004
    # s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    # e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)  # second differentiation
    # e2 = time.time()
    # print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv

######## LiSSA ########

def get_inverse_hvp_lissa(v, model, device, param_influence, train_loader, 
                          damping, epochs, recursion_depth, scale=1e4, save_dir="./", seed=100):
    
    logger = logging.getLogger("hvp_lissa")
    os.makedirs(save_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{seed}.txt"))
    logger.addHandler(handler)

    ihvp = None
    v = [w.clone().cuda() for w in v]
    logger.info(f"norm of v: {np.linalg.norm(gather_flat_grad(v).cpu().numpy())}")
    for i in range(epochs):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in tqdm(range(1,recursion_depth+1), desc="Calculating HVP", position=0):
            try:
                data = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                data = next(lissa_data_iterator)
                
            model.eval()

            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            tokens = {"input_ids":ids, "attention_mask":mask}
            
            model.zero_grad()
            outputs = model(tokens)
            logits = outputs
            loss = F.cross_entropy(logits, y, reduction='mean')
            
            hvp = hv(loss, param_influence, cur_estimate)

            if j % 100 == 0 or j == 1:
                logger.info(f"  epoch {i} step {j}. hvp norm: {np.linalg.norm(gather_flat_grad(hvp).cpu().numpy())}")
                logger.info(f"  epoch {i} step {j}. cur_est norm: {np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())}")
                grad_change = gather_flat_grad([_a + (0 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]).cpu().numpy()
                logger.info(f"  epoch {i} step {j}. grad change: {np.linalg.norm(grad_change)}")
            
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            
        
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
        logger.info(f"end epoch {i}, ihvp norm:{np.linalg.norm(gather_flat_grad(ihvp).cpu().numpy())}")
            
    # return_ihvp = gather_flat_grad(ihvp) # flatten HVP to a vector, NOT compatible to calc_influence_with_hvp.py script
    # return_ihvp /= epochs
    ihvp = [a / epochs for a in ihvp]
    return ihvp


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
    # lissa
    group_model.add_argument("--lissa_depth", default=1.0, type=float, required=False,
                    help="lissa_depth*lan(train_loader) = num of iterations to update HVP estimate")

    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--seed", default=100, type=int, required=False,
                    help="random seed")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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
    max_length = 30

    dev_dataset = CKBPDataset(infer_file[infer_file["split"] == "dev"], tokenizer, max_length, sep_token=sep_token) 
    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)

    # training
    train_dataset = pd.read_csv(args.train_file_path)
    train_params = {
            'batch_size': 32,
            'num_workers': 1
    }

    training_set = CKBPDataset(train_dataset, tokenizer, max_length, sep_token=sep_token)
    training_sampler = RandomSampler(training_set,
                                      replacement=True,
                                      num_samples=args.num_train)

    training_loader = DataLoader(training_set, **train_params, sampler=training_sampler, drop_last=True)

    # calculate grad w.r.t the whole dev set
    grad = get_validation_grad(model, dev_dataloader)

    # for name, p in model.named_parameters():
    #     print(name)
    param_influence = [p for name, p in model.named_parameters() if not name.startswith("model.pooler.dense")]

    save_path = os.path.join(os.path.dirname(args.checkpoint_path),
        f"hvp_lissa_{args.num_train}_damping_{args.damping}_c_{args.c}_epoch_{args.epoch}", 
    )

    HVP = get_inverse_hvp_lissa(grad, model, args.device, param_influence, 
        training_loader, damping=args.damping, epochs=args.epoch, 
        recursion_depth=int(len(training_loader)*args.lissa_depth), scale=args.c,
        save_dir=save_path, seed=args.seed)

    torch.save(HVP,
        save_path + f"/seed_{args.seed}.pth"
    )

if __name__ == "__main__":
    main()