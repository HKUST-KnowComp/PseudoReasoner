# Importing stock libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import sys
sys.path.append(os.getcwd())
import logging

logger = logging.getLogger()

logger.setLevel(logging.INFO)


class CKBPDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length, sep_token=" ", model="kgbert", is_eval=False):
        # infer file dataset given a certain relation
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index()
        self.max_length = max_length
        self.text = self.data["head"]
        self.ctext = self.data["tail"]
        self.rtext = self.data["relation"]
        if "label" in self.data:
            self.labels = self.data["label"]
        else:
            # useful when scoring pseudo edges.
            self.labels = pd.Series([0 for i in range(len(self.data))]) 
        if "soft_label" in self.data:
            self.soft_label = self.data["soft_label"]
        else:
            self.soft_label = pd.Series([0 for i in range(len(self.data))]) 
        self.sep_token = sep_token
        self.model = model
        self.is_eval = is_eval

        # for evaluation set only
        if "class" in self.data:
            self.clss = self.data["class"]
        else:
            logger.warning("No class labels in the dataset")
            self.clss = pd.Series(["" for i in range(len(self.data))]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        rtext = str(self.rtext[index])
        rtext = ' '.join(rtext.split())

        if self.model == "kgbert":
            source = self.tokenizer.batch_encode_plus([text + self.sep_token + rtext + self.sep_token + ctext], 
                padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        elif self.model == "gpt2":
            source = self.tokenizer.batch_encode_plus([text + ' ' + rtext + ' ' + ctext + ' [EOS]'], padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
        elif self.model == "comet_gpt2":
            if not self.is_eval:
                source = self.tokenizer.batch_encode_plus([text + ' ' + rtext + ' [GEN] ' + ctext + ' [EOS]'], padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True)
            else:
                source = self.tokenizer.batch_encode_plus([text + ' ' + rtext + ' [GEN]'], padding=False, max_length=self.max_length, return_tensors='pt', truncation=True)

        # if index < 5:
        #     logger.info("Source: {}".format(self.tokenizer.batch_decode(source['input_ids'])))

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long), 
            'label': torch.tensor(self.labels[index]).to(dtype=torch.long),
            'soft_label': torch.tensor(self.soft_label[index]).to(dtype=torch.float32),
            'clss': self.clss[index],
        }
