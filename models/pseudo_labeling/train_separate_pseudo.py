import time
import random
import argparse
import os, sys, shutil
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer

from models.pseudo_labeling.model import KGBERTClassifier
from models.model_utils import evaluate
from models.dataloader import CKBPDataset
from utils.ckbp_utils import special_token_list

import logging
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
    group_trainer.add_argument("--lr", default=1e-5, type=float, required=False,
                    help="learning rate")
    group_trainer.add_argument("--pseudo_lr", default=1e-5, type=float, required=False,
                    help="learning rate")
    group_trainer.add_argument("--lrdecay", default=1, type=float, required=False,
                        help="the learning rate decay (every x steps)")
    group_trainer.add_argument("--pseudo_lrdecay", default=1, type=float, required=False,
                        help="the learning rate decay (every x steps)")
    group_trainer.add_argument("--decay_every", default=500, type=int, required=False,
                    help="afer every x steps, decay the lr")
    group_trainer.add_argument("--pseudo_decay_every", default=500, type=int, required=False,
                    help="afer every x steps, decay the lr")
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                        help="test batch size")
    group_trainer.add_argument("--epochs", default=3, type=int, required=False,
                        help="the number of epochs to train model on labeled data")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on labeled data. used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=30, type=int, required=False,
                        help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_metric", type=str, required=False, default="overall_auc",
                    choices=["grouped_auc", "overall_auc", "accuracy"],
                    help="evaluation metric.")
    group_trainer.add_argument("--eval_every", default=250, type=int, required=False,
                        help="eval on test set every x steps.")
    group_trainer.add_argument("--pseudo_eval_every", default=250, type=int, required=False,
                        help="eval on test set every x steps during pseudo pretraining.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                        help="whether to use special token to represent relation.")
    group_trainer.add_argument("--noisy_training", action="store_true",
                        help="whether to have a noisy training, flip the labels with probability p_noisy.")
    group_trainer.add_argument("--p_noisy", default=0.0, type=float, required=False,
                    help="probability to flip the labels")

    # pseudo-label-related args
    group_pseudo_label = parser.add_argument_group("pseudo label configs")
    group_pseudo_label.add_argument("--pseudo_label", action="store_true",
                    help="whether to include pseudo labels for training")
    group_pseudo_label.add_argument("--pseudo_examples_path", required=False,
                        help="paths to the csv file containing pseudo-labels")
    group_pseudo_label.add_argument("--pretrain_pseudo_epochs", required=False, default=1, type=int,
                        help="num of epochs pretraining model using pseudo examples.")
    group_trainer.add_argument("--pretrain_pseudo_steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on pseudo labeled data. used for the case training model less than 1 epoch")
    ## TODO: may change to step, in order to see the learning curve of model on pseudo labelled data

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
    group_data.add_argument("--debug", action="store_true",
                    help="whether to print some debug messages")
    group_data.add_argument("--resume_train_from_best_pseudo", action="store_true",
                    help="whether to retrieve the best pseudo checkpoint and start finetuning on the labelled dataset.")
    group_data.add_argument("--show_result_only", action="store_true",
                    help="useful when tuning parameters and don't need to store data. Will clear checkpoints and tokenizers."
                         "will output a line containing the final result.")

    args = parser.parse_args()
    return args

def main():
    # get all arguments
    args = parse_args()
    
    experiment_name = args.experiment_name
    if args.noisy_training:
        experiment_name = experiment_name + f"_noisy_{args.p_noisy}"

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm), 
        f"bs{args.batch_size}", f"evalstep{args.eval_every}"]),  # backbone params
        f"pseudo_lr{args.pseudo_lr}_decay{args.pseudo_lrdecay}_step{args.pseudo_decay_every}_evalstep{args.pseudo_eval_every}", # pseudo params
        experiment_name, # additional experiment name
    ) 
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

    # load data
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


    ## pseudo label dataset
    pseudo_dataset = pd.read_csv(args.pseudo_examples_path)
    pseudo_training_set = CKBPDataset(pseudo_dataset, tokenizer, args.max_length, sep_token=sep_token)
    pseudo_loader = DataLoader(pseudo_training_set, **train_params, drop_last=True)


    # model training
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    best_epoch, best_iter = 0, 0
    best_val_score = 0
    early_steps = [1,3,10,30, 50, 75, 100, 125, 150, 175, 200]

    phases_args = {'pseudo': (args.pretrain_pseudo_epochs, args.pretrain_pseudo_steps, pseudo_loader, args.eval_every),
                'labelled': (args.epochs, args.steps, training_loader, args.pseudo_eval_every)}

    ## training
    ## separate Pseudo Label Pretraining. use for-loop to reuse code.
    for phase in ['pseudo', 'labelled']:
        num_epochs, num_steps, data_loader, eval_every = phases_args[phase]
        if phase == "labelled":
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            if args.resume_train_from_best_pseudo:
                model.load_state_dict(torch.load(save_dir + f"/best_model_seed_{args.seed}.pth"))
            
        if phase == "pseudo":
            for g in optimizer.param_groups:
                g['lr'] = args.pseudo_lr
            if args.pseudo_lrdecay < 1:
                pseudo_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.pseudo_lrdecay)

        for e in range(num_epochs):
            if args.show_result_only:
                iterator = enumerate(data_loader, 1)
            else:
                iterator = tqdm(enumerate(data_loader, 1), total=len(data_loader))
            for iteration, data in iterator:
                # the iteration starts from 1. 

                y = data['label'].to(args.device, dtype=torch.long)
                # noisy training
                if args.noisy_training:
                    noisy_vec = torch.rand(len(y))
                    y = y ^ (noisy_vec < args.p_noisy) 
                    # flip label with probability p_noisy

                ids = data['ids'].to(args.device, dtype=torch.long)
                mask = data['mask'].to(args.device, dtype=torch.long)
                tokens = {"input_ids":ids, "attention_mask":mask}

                logits = model(tokens)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if phase == "pseudo" and args.pseudo_lrdecay < 1 and iteration % args.pseudo_decay_every == 0:
                    pseudo_lr_scheduler.step()


                if (eval_every > 0 and iteration % eval_every == 0) or \
                    iteration in early_steps:
                    model.eval()

                    # validation auc
                    eval_auc, _ = evaluate(tokenizer, model, args.device, dev_dataloader)
                    assert _ == len(dev_dataset)

                    if eval_auc > best_val_score:
                        best_val_score = eval_auc
                        if args.save_best_model or args.resume_train_from_best_pseudo:
                            torch.save(model.state_dict(), save_dir + f"/best_model_seed_{args.seed}.pth")
                            tokenizer.save_pretrained(save_dir + "/best_tokenizer")
                        
                        best_epoch, best_iter = e, iteration
                        logger.info(f"Best validation score {best_val_score} reached at phase '{phase}' epoch {best_epoch} step {best_iter}")

                        # calc test scores after every x step
                        tst_auc, _, class_scores, _ = evaluate(tokenizer, model, args.device, tst_dataloader, class_break_down=True)

                        logger.info(f"Overall auc & Test Set & CSKB Head + ASER tail & ASER edges.")
                        logger.info(f"test scores at phase '{phase}' epoch {e} step {iteration}:" + " & ".join([str(round(tst_auc*100, 1))]+\
                                [str(round(class_scores[clss]*100, 1)) for clss in ["test_set", "cs_head", "all_head"]]) )
                    elif args.debug:
                        # calc test scores after every x step
                        tst_auc, _, class_scores, _ = evaluate(tokenizer, model, args.device, tst_dataloader, class_break_down=True)

                        logger.info(f"Overall auc & Test Set & CSKB Head + ASER tail & ASER edges.")
                        logger.info(f"test scores at phase '{phase}' epoch {e} step {iteration}:" + " & ".join([str(round(tst_auc*100, 1))]+\
                                [str(round(class_scores[clss]*100, 1)) for clss in ["test_set", "cs_head", "all_head"]]) )                        

                    model.train()

                if num_steps > 0 and iteration >= num_steps:
                    break
    if args.show_result_only:
        os.remove(save_dir + f"/best_model_seed_{args.seed}.pth")
        shutil.rmtree(save_dir + "/best_tokenizer")


if __name__ == "__main__":
    main()
