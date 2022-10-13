import os
import json
import pandas as pd
import numpy as np
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
import argparse

def read_jsonl_lines(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def get_gen(strs):
    strs = strs.split()
    st = 0
    ed = 0
    for i in range(len(strs)):
        if strs[i] == "[GEN]":
            st = i
        if strs[i] == "[EOS]":
            ed = i
            break
    return " ".join(strs[st+1:ed])

def main():

    config = argparse.Namespace()
    config.PRED_RESULTS_FILE = os.environ.get("PRED_RESULTS_FILE", "") # the prediction jsonl
    config.GROUND_CSV = os.environ.get("GROUND_CSV", "") # the ground csv file.

    df = pd.read_csv(config.GROUND_CSV)
    preds = read_jsonl_lines(config.PRED_RESULTS_FILE)
    grounds = [json.loads(tails) for tails in df["tail"]]


    generations = [get_gen(preds[i]['generations'][0]) for i in range(len(preds))]

    hyps = {idx: [strippedlines] for (idx, strippedlines) in enumerate(generations)}
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(grounds)}
    print(" | ".join([str(round(score, 10)) for score in Bleu(4).compute_score(refs, hyps)[0]]))

if __name__ == '__main__':

    main()