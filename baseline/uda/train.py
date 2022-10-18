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
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F

from models.pseudo_labeling.model import KGBERTClassifier
from models.model_utils import evaluate
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer

from utils.ckbp_utils import special_token_list

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str, 
                        required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_from_path", required=False, default="",
                        help="pretrain this model from a checkpoint") # a bit different from --ptlm.

    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                        help="device")
    group_trainer.add_argument("--optimizer", default='ADAM', type=str, required=False,
                        help="optimizer")
    group_trainer.add_argument("--lr", default=0.01, type=float, required=False,
                        help="learning rate")
    group_trainer.add_argument("--lrdecay", default=1, type=float, required=False,
                        help="learning rate decay every x steps")
    group_trainer.add_argument("--decay_every", default=500, type=int, required=False,
                        help="show test result every x steps")
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                        help="test batch size")
    group_trainer.add_argument("--epochs", default=3, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on labeled data. "
                        "used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=30, type=int, required=False,
                        help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_metric", type=str, required=False, default="overall_auc",
                        choices=["grouped_auc", "overall_auc", "accuracy"],
                        help="evaluation metric.")
    group_trainer.add_argument("--eval_every", default=250, type=int, required=False,
                        help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                        help="whether to use special token to represent relation.")
    
    # IO-related
    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="results",
                        type=str, required=False,
                        help="where to output.")
    group_data.add_argument("--train_csv_path", default='', type=str, required=True)
    group_data.add_argument("--evaluation_file_path", default="data/evaluation_set.csv", 
                        type=str, required=False)    
    group_data.add_argument("--model_dir", default='models', type=str, required=False,
                        help="Where to save models.")
    group_data.add_argument("--save_best_model", action="store_true",
                        help="whether to save the best model.")
    group_data.add_argument("--log_dir", default='logs', type=str, required=False,
                        help="Where to save logs.")
    group_data.add_argument("--experiment_name", default='', type=str, required=False,
                        help="A special name that will be prepended to the dir name of the output.")
    
    group_data.add_argument("--seed", default=401, type=int, required=False,
                        help="random seed")

    # UDA
    group_uda = parser.add_argument_group("UDA related configs")
    group_uda.add_argument("--unsup_data_dir", default=" ",
                        type=str, required=True,
                        help="path to unsup dataa and its tfidf-augmented version")
    group_uda.add_argument("--aug_ops", default='tf_idf-0.1', 
                        type=str, required=True)
    group_uda.add_argument("--unsup_ratio", default=0.5, 
                        type=float, required=True,
                        help="ratio of original unsup data w.r.t sup data."
                        "Eg. input = 0.x, then in a batch, there are batch_size*0.x"
                        "samples from ori. unsup and tfidf-aug. unsup set")
    group_uda.add_argument("--uda_coeff", default=1.0, 
                        type=float, required=True,
                        help="coefficient of uda loss in total loss")

    # for text, these hyperparams are not set (refer to uda/text/main.py)
    # so not used =))
    group_uda.add_argument("--uda_softmax_temp", default=-1, 
                        type=float, required=False,
                        help="temperature in softmax")
    group_uda.add_argument("--uda_confidence_thresh", default=-1, 
                        type=float, required=False,
                        help="only calculate loss with unsup pairs with max logits > this threashold")

    args = parser.parse_args()
    return args


def main():
    # get all arguments
    args = parse_args()

    save_dir = os.path.join(args.output_dir, "_".join([args.ptlm, f"bs{args.batch_size}", f"evalstep{args.eval_every}"])+args.experiment_name )
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("kg-bert")
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
    logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    print('loading model ...')
    model = KGBERTClassifier(args.ptlm).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.ptlm)
    sep_token = tokenizer.sep_token
    if args.relation_as_special_token:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
        model.model.resize_token_embeddings(len(tokenizer))
    
    if args.pretrain_from_path:
        model.load_state_dict(torch.load(args.pretrain_from_path))
    print('finished loading model ...')

    # load data
    print('loading data ...')
    train_dataset = pd.read_csv(args.train_csv_path)
    infer_file = pd.read_csv(args.evaluation_file_path)
    
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    training_set = CKBPDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token)
    training_loader = DataLoader(training_set, **train_params, drop_last=True)

    dev_dataset = CKBPDataset(infer_file[infer_file["split"] == "dev"], tokenizer, args.max_length, sep_token=sep_token) 
    tst_dataset = CKBPDataset(infer_file[infer_file["split"] == "tst"], tokenizer, args.max_length, sep_token=sep_token) 

    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)
    tst_dataloader = DataLoader(tst_dataset, **val_params, drop_last=False)

    # pseudo label dataset
    unsup_dataset = pd.read_csv(args.unsup_data_dir)
    unsup_aug_dataset = pd.read_csv(args.unsup_data_dir[:-4]+'_'+args.aug_ops+'.csv')
    unsup_aug_dataset["relation"] = ''
    unsup_aug_dataset["tail"] = ''

    unsup_params = {
        'batch_size': max(int(args.batch_size*args.unsup_ratio), 1),
        'shuffle': True,
        'num_workers': 5
    }

    unsup_set = CKBPDataset(unsup_dataset, tokenizer, args.max_length, sep_token=' ')
    unsup_aug_set = CKBPDataset(unsup_aug_dataset, tokenizer, args.max_length, sep_token=' ')

    unsup_loader = DataLoader(unsup_set, **unsup_params, drop_last=True)
    unsup_aug_loader = DataLoader(unsup_aug_set, **unsup_params, drop_last=True)
    print('finished loading data ...')


    # model training
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    unsup_criterion = torch.nn.KLDivLoss(reduction="batchmean")
    best_epoch, best_iter = 0, 0
    best_val_score = 0
    model.train()
    early_steps = [1,3,10,30, 50, 75, 100, 125, 150, 175, 200]
    
    if args.steps > 0:
        args.epochs = 1

    for e in range(args.epochs):
        aggregated = zip(training_loader, unsup_loader, unsup_aug_loader)
        for iteration, (ldata, uodata, uadata) in \
            tqdm(enumerate(aggregated, 1), total=args.steps):
            # the iteration starts from 1. 

            # sup training
            y = ldata['label'].to(args.device, dtype=torch.long)
            ids = ldata['ids'].to(args.device, dtype=torch.long)
            mask = ldata['mask'].to(args.device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}
            logits = model(tokens)
            sup_loss = criterion(logits, y)

            # unsup training
            unsup_logits = {}
            for s, data in zip(['ori', 'aug'], [uodata, uadata]): # unsup original vs augmented data
                y = data['label'].to(args.device, dtype=torch.long)
                ids = data['ids'].to(args.device, dtype=torch.long)
                mask = data['mask'].to(args.device, dtype=torch.long)
                tokens = {"input_ids":ids, "attention_mask":mask}
                unsup_logits[s] = F.softmax(model(tokens), dim=-1)
            
            # follow the formula implied by code of uda.py
            unsup_loss = unsup_criterion(torch.log(unsup_logits['aug']), unsup_logits['ori'])

            # backward propagation
            loss = (sup_loss + args.uda_coeff*unsup_loss)/(1+args.uda_coeff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del sup_loss, unsup_loss, loss, y, ids, mask

            if (args.eval_every > 0 and iteration % args.eval_every == 0) or \
                iteration in early_steps:
                model.eval()

                # validation auc
                eval_auc, _ = evaluate(tokenizer, model, args.device, dev_dataloader)
                assert _ == len(dev_dataset)

                if eval_auc > best_val_score:
                    best_val_score = eval_auc
                    if args.save_best_model:
                        torch.save(model.state_dict(), save_dir + f"/best_model_seed_{args.seed}.pth")
                        tokenizer.save_pretrained(save_dir + "/best_tokenizer")
                    
                    best_epoch, best_iter = e, iteration
                    logger.info(f"Best validation score reached at epoch {best_epoch} step {best_iter}")

                # calc test scores after every x step
                tst_auc, _, class_scores, _ = evaluate(tokenizer, model, args.device, tst_dataloader, class_break_down=True)

                logger.info(f"Overall auc & Test Set & CSKB Head + ASER tail & ASER edges")
                logger.info(f"test scores at epoch {e} step {iteration}:" + " & ".join([str(round(tst_auc*100, 1))]+\
                        [str(round(class_scores[clss]*100, 1)) for clss in ["test_set", "cs_head", "all_head"]]) )

                model.train()
            
            if args.steps > 0 and iteration >= args.steps:
                break

if __name__ == "__main__":
    main()
