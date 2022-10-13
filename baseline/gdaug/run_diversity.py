import os
import csv
from tqdm import tqdm
from tqdm import trange
import re
import argparse
import pandas as pd

def tokenize(str):
    str = re.findall(r"[\w']+|[.,!?;]", str)

    return " ".join(str)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file_path",
    default= "",
    type=str,
    help=
    "The input data dir. "
)


parser.add_argument(
    "--sample_size",
    default=100000 ,
    type=int,
    help=
    "num samples."
)

parser.add_argument(
    "--label",
    required=True,
    type=int,
    help=
    "label. 0 or 1"
)

args = parser.parse_args()
sample_size = args.sample_size
label = args.label
input_file_path = args.input_file_path

df_train = pd.read_csv(input_file_path)
df_train = df_train[df_train["label"] == label]
df_train = df_train.sample(frac=1).reset_index(drop=True)

data_idx = []
vocab = []

print("build vocab")

for i, (head, relation, tail) in tqdm(df_train[["head", "relation", "tail"]].iterrows()):
    data_idx.append(i)
    line = " ".join([str(head), relation, str(tail)])
    
    vocab_set = tokenize(line).split(" ")
    
    vocab_set = set(vocab_set)
    vocab.append(vocab_set)

print(len(data_idx))


selected = [False] * len(data_idx)
selected_vocab = set([])
output_idx = []
previous_vocab_increase = [None] * len(data_idx)


for i in trange(sample_size):
    max_vocab_increase = -1
    max_idx = -1
    for j, (s,v,p_v_i) in enumerate( zip(selected,vocab,previous_vocab_increase) ):
        if s:
            continue
        if p_v_i is not None:
            if p_v_i <= max_vocab_increase:
                continue

        vocab_increase = len(v - selected_vocab)
        previous_vocab_increase[j] = vocab_increase

        if vocab_increase > max_vocab_increase:
            max_idx = j
            max_vocab_increase = vocab_increase
    if max_idx == -1:
        o = 2
        print("error")
    else:
        #print(max_vocab_increase)
        output_idx.append(data_idx[max_idx])
        selected_vocab.update(vocab[max_idx])
        selected[max_idx] = True

df_train.loc[output_idx].reset_index(drop=True).to_csv(input_file_path[:-4] + f"_div_{sample_size}_label_{label}.csv", index=False)