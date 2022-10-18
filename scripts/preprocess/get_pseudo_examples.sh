RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 \
    RELATION=xWant \
    HARD_THRESHOLD=1 THRESHOLD="4.36|5.8|4.0|3.6|4.0" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/test.csv \
    python ./data_preparation/filter_and_select.py

RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 \
    HARD_THRESHOLD=1 THRESHOLD="4.36|5.8|4.0|3.6|4.0" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/pseudo_triple_maxproportion_0.333_new.csv \
    python ./data_preparation/filter_and_select.py

RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/kgbert_base_rel_special \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    INPUT_NEGATIVE_LOSS=0 \
    HARD_THRESHOLD=1 THRESHOLD="4.36|5.8|0.1|0.1|0.1" \
    OUTPUT_FILE=./data/pseudo_label_trn/kgbert_base_rel_special/pseudo_triple_maxproportion_0.333_new.csv \
    python ./data_preparation/filter_and_select.py


## aggregate data for ablation study on filtering
# no FILTER_BY_FLUENCY_LOSS, no constraint in the number of input triples
RELATION=atomic18 \
    FILTER_BY_FLUENCY_LOSS=0 \
    MAX_PROPORTION=0.1    \
    INPUT_NEGATIVE_LOSS=1     \
    PSEUDO_LABEL_SCORE_DIR="./data/pseudo_label_scores/ckbp_gpt2xl_eval10"    \
    FLUENCY_LOSS_DIR="./data/pseudo_label_scores/gpt2xl_zeroshot"   \
    OUTPUT_FILE="./data/pseudo_label_trn/ckbp_gpt2xl_eval10_gpt2xl_zeroshot/pseudo_triple_atomic18_only_cms_filter_maxprop0.1.csv" \
    python ./data_preparation/filter_and_select.py

# no FILTERs, no constraint in the number of input triples
RELATION=atomic18 \
    FILTER_BY_FLUENCY_LOSS=0 \
    FILTER_BY_PSEUDO_LABEL_SCORE=0 \
    MAX_PROPORTION=0.05    \
    INPUT_NEGATIVE_LOSS=1     \
    PSEUDO_LABEL_SCORE_DIR="./data/pseudo_label_scores/ckbp_gpt2xl_eval10"    \
    FLUENCY_LOSS_DIR="./data/pseudo_label_scores/gpt2xl_zeroshot"   \
    OUTPUT_FILE="./data/pseudo_label_trn/ckbp_gpt2xl_eval10_gpt2xl_zeroshot/pseudo_triple_atomic18_no_filter_maxprop0.05.csv" \
    python ./data_preparation/filter_and_select.py