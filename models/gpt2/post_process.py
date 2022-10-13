import os
import json
import pandas as pd
import numpy as np
import argparse

all_relations = [
        "xWant", "oWant", "general Want",
        "xEffect", "oEffect", "general Effect",
        "xReact", "oReact", "general React",
        "xAttr",
        "xIntent",
        "xNeed",
        "Causes", "xReason",
        "isBefore", "isAfter",
        'HinderedBy',
        'HasSubEvent',
    ]

def split_head_relation(source):
    tokens = source.split()
    if tokens[-1] != '[GEN]':
        return
    else:
        if tokens[-2] in all_relations:
            return " ".join(tokens[:-2]), tokens[-2]
        elif tokens[-3] + " " + tokens[-2] in all_relations:
            return " ".join(tokens[:-3]), tokens[-3] + " " + tokens[-2]
        else:
            return

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
    pred_file = os.environ.get("PRED_RESULTS_FILE", "") # the prediction jsonl
    k = int(os.environ.get("K", 1)) # number of beam generations to keep
    out_csv = os.environ.get("OUT_CSV", "") # the ground csv file.

    preds = read_jsonl_lines(pred_file)
    generations = [[get_gen(preds[i]['generations'][j]) for j in range(k)] for i in range(len(preds))]



if __name__ == '__main__':

    main()