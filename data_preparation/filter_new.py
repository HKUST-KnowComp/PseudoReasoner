import sys, os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import argparse

from utils.ckbp_utils import (
    atomic18_relations, 
    num_of_triples_emnlp21,
)

rel_dict = {"gWant": "general Want", "gEffect":"general Effect", "gReact":"general React"}

def filter_pseudo_label_hard_threshold(df_pseudo_labels, config, threshold_this_rel):
    

    if config.INPUT_NEGATIVE_LOSS:  # GPT2-loss. less than threshold.
        df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['pseudo_label_score'].lt(threshold_this_rel['pseudo_label_score']["up"])].sort_values(by='pseudo_label_score', ascending=False).reset_index(drop=True)
        # descending.
        if sum(df_pseudo_labels['pseudo_label_score'].gt(threshold_this_rel['pseudo_label_score']["low"])) > config.MIN_LEFT_AFTER_CMS_LOSS:
            df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['pseudo_label_score'].gt(threshold_this_rel['pseudo_label_score']["low"])]
        else:
            df_pseudo_labels = df_pseudo_labels[:config.MIN_LEFT_AFTER_CMS_LOSS]
    else:   # KGBert-logit. larger than threshold.
        df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['pseudo_label_score'].gt(threshold_this_rel['pseudo_label_score']["low"])].sort_values(by='pseudo_label_score', ascending=True).reset_index(drop=True)
        # ascending
        if sum(df_pseudo_labels['pseudo_label_score'].lt(threshold_this_rel['pseudo_label_score']["up"])) > config.MIN_LEFT_AFTER_CMS_LOSS:
            df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['pseudo_label_score'].lt(threshold_this_rel['pseudo_label_score']["up"])]
        else:
            df_pseudo_labels = df_pseudo_labels[:config.MIN_LEFT_AFTER_CMS_LOSS]
    
    return df_pseudo_labels


if __name__ == '__main__':
    config = argparse.Namespace()
    
    # control the number of triples after filtering
    config.MIN_LEFT_AFTER_FLUENCY_LOSS = int(os.environ.get("MIN_LEFT_AFTER_FLUENCY_LOSS", 100000))
    config.MIN_LEFT_AFTER_CMS_LOSS = int(os.environ.get("MIN_LEFT_AFTER_CMS_LOSS", 20000))
    config.NUM_SAMPLES_DEBUG = int(os.environ.get('NUM_SAMPLES_DEBUG', -1))

    # ablation study on filtering
    config.FILTER_BY_FLUENCY_LOSS = int(os.environ.get('FILTER_BY_FLUENCY_LOSS', 1)) # set the as 1 to enable
    config.FILTER_BY_PSEUDO_LABEL_SCORE = int(os.environ.get('FILTER_BY_PSEUDO_LABEL_SCORE', 1)) # set the as 1 to enable

    # i/o related
    config.INPUT_NEGATIVE_LOSS = int(os.environ.get('INPUT_NEGATIVE_LOSS', "1")) # 0 for KG-BERT, 1 for GPT2 loss
    config.PSEUDO_LABEL_SCORE_DIR = os.environ.get('PSEUDO_LABEL_SCORE_DIR', "") # previously known as CMS_LOSS_DIR
    config.FLUENCY_LOSS_DIR = os.environ.get('FLUENCY_LOSS_DIR', "") # try comet as fluency filter later?
    config.CANDIDATE_PATH = os.environ.get('CANDIDATE_PATH', "") # the path to provide triples.
    config.OUTPUT_FILE = os.environ.get('OUTPUT_FILE', f"pseudo_triple.csv")

    # config.THRESHOLD = os.environ.get('THRESHOLD', '4.36|5.8|4.0|3.6|4.0') # the default threshold
    config.THRESHOLD = os.environ.get('THRESHOLD', '4.36|5.8|4.0|3.7|2.8|2.0') # the default threshold
    # fluency_upper|fluency_lower|T_min^-|T_max^-|T_min^+|T_max^+

    threshold = {
        'fluency': {
            'low': float(config.THRESHOLD.split("|")[0]),
            'up': float(config.THRESHOLD.split("|")[1])
        },
        'pseudo_label_score': {
            'neg_low':float(config.THRESHOLD.split("|")[2]),
            'neg_up':float(config.THRESHOLD.split("|")[3]),
            'pos_low':float(config.THRESHOLD.split("|")[4]),
            'pos_up':float(config.THRESHOLD.split("|")[5]),
        }
    }


    rel_list = list(num_of_triples_emnlp21.keys())
    all_selected = pd.DataFrame()

    for r in rel_list:
        print('Considering candidates for relation {}'.format(r))

        # deal with the case that fluency and cms losses are stored in two different files. 
        # DISCLAIMER: the order of samples in those two files must be the same
        if config.FLUENCY_LOSS_DIR != "" and config.PSEUDO_LABEL_SCORE_DIR != "":
            df_fluency = pd.read_csv(os.path.join(config.FLUENCY_LOSS_DIR, r + '.csv'))
            df_pseudo_labels = pd.read_csv(os.path.join(config.PSEUDO_LABEL_SCORE_DIR, r + '.csv'))
            df_pseudo_labels.rename(inplace=True, columns={
                'score': 'pseudo_label_score'
            })
            df_pseudo_labels['fluency_loss'] = df_fluency['score'].tolist()
            # negative by default
            df_pseudo_labels['fluency_loss'] = -df_pseudo_labels['fluency_loss']

            if config.INPUT_NEGATIVE_LOSS: # they should be gpt2 loss stored with negative scores.
                df_pseudo_labels['pseudo_label_score'] = -df_pseudo_labels['pseudo_label_score']

            # del df_fluency
            print('Merging two files')
            print(df_pseudo_labels.tail(3))
        else:
            raise NotImplementedError

        # the path to define all the candidate triples 
        # The order should be exactly the same.
        if config.CANDIDATE_PATH != "":
            candidate_triple_df = pd.read_csv(os.path.join(config.CANDIDATE_PATH, r + '.csv'))
            # lengths are the same
            assert len(df_pseudo_labels) == len(candidate_triple_df)
            df_pseudo_labels["head"] = candidate_triple_df["head"]
            df_pseudo_labels["tail"] = candidate_triple_df["tail"]
            df_pseudo_labels["relation"] = candidate_triple_df["relation"]

        
        if config.NUM_SAMPLES_DEBUG > 0:
            print('Processing first {} samples'.format(config.NUM_SAMPLES_DEBUG))
            df_pseudo_labels = df_pseudo_labels[:config.NUM_SAMPLES_DEBUG]

        # prepare threshold:


        if config.INPUT_NEGATIVE_LOSS:
            # if it's GPT2-loss, then the upper bound indicates the worst pseudo example.
            pseudo_score_up = threshold['pseudo_label_score']['neg_low']
            pseudo_score_low = threshold['pseudo_label_score']['pos_up']
        else:
            # if it's KG-BERT logit scores, then the lower bound indicates the worst pseudo example.
            pseudo_score_up = 1
            pseudo_score_low = threshold['pseudo_label_score'].get(r[0], threshold['pseudo_label_score']["others"])
            
        
        threshold_this_rel = {
            'fluency': {
                'low': threshold['fluency']['low'],
                'up': threshold['fluency']['up'],
            },
            'pseudo_label_score': {
                'low': pseudo_score_low,
                'up': pseudo_score_up,
            }
        }


        ## filter by fluency_loss
        # df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['fluency_loss'].between(threshold['fluency']['low'], threshold['fluency']['up'])]
        if config.FILTER_BY_FLUENCY_LOSS:
            print('Filtering by fluency loss')
            df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['fluency_loss'] >threshold_this_rel['fluency']['low']].sort_values(by='fluency_loss', ascending=True).reset_index(drop=True)
            if sum((df_pseudo_labels['fluency_loss']<threshold_this_rel['fluency']['up']).astype('int32')) < config.MIN_LEFT_AFTER_FLUENCY_LOSS:
                df_pseudo_labels = df_pseudo_labels[:config.MIN_LEFT_AFTER_FLUENCY_LOSS]
            else:
                df_pseudo_labels = df_pseudo_labels[df_pseudo_labels['fluency_loss']<threshold_this_rel['fluency']['up']]

        df_pseudo_labels.drop(['fluency_loss'], axis=1, inplace=True)

        if config.INPUT_NEGATIVE_LOSS: #GPT2-loss
            df_pseudo_labels.sort_values(by='pseudo_label_score', ascending=True, inplace=True)
            # larger indicate poorer commonsense. Ascending
        else: #KG-BERT logit scores
            df_pseudo_labels.sort_values(by='pseudo_label_score', ascending=False, inplace=True)
            # larger indicate better commonsense. Descending

        df_pseudo_labels.reset_index(drop=True, inplace=True)

        if config.FILTER_BY_PSEUDO_LABEL_SCORE:
            print('Filtering by pseudo label scores (either gpt2 loss of kg-bert logit score)')

            df_pseudo_labels = filter_pseudo_label_hard_threshold(df_pseudo_labels, config, threshold_this_rel)

            pos_idx = (df_pseudo_labels["pseudo_label_score"] >= threshold['pseudo_label_score']['pos_up']) & \
                        (df_pseudo_labels["pseudo_label_score"] <= threshold['pseudo_label_score']['pos_low']) 
            df_pos = df_pseudo_labels[pos_idx]
            df_pos = df_pos.sort_values(by='pseudo_label_score', ascending=True).reset_index(drop=True)
            df_pos["label"] = 1

            neg_idx = (df_pseudo_labels["pseudo_label_score"] >= threshold['pseudo_label_score']['neg_up']) & \
                        (df_pseudo_labels["pseudo_label_score"] <= threshold['pseudo_label_score']['neg_low']) 
            df_neg = df_pseudo_labels[neg_idx]
            df_neg = df_neg.sort_values(by='pseudo_label_score', ascending=False).reset_index(drop=True)            
            df_neg["label"] = 0

            num_triples = num_of_triples_emnlp21[r] // 2
            df_pos = df_pos[:min(num_triples, len(df_pos))]
            df_neg = df_neg[:min(num_triples, len(df_neg))]

            selected_examples = pd.concat([df_pos, df_neg]).reset_index(drop=True)

            print(df_pos.head(5))
            print(df_pos.tail(5))
            print(df_neg.head(5))
            print(df_neg.tail(5))


        num_triples_remaining = df_pseudo_labels['pseudo_label_score'].size
        print("After filtering, {} candidates left".format(num_triples_remaining))

        
        selected_examples["relation"] = rel_dict.get(r, r)

        all_selected = all_selected.append(selected_examples)

    os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
    all_selected.to_csv(config.OUTPUT_FILE, index=False)

