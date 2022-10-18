

# default proportion thresholds: 0.005|0.3|0.1|0.6
# for xWant, the corresponding low/up hard thresh are: 4.815|5.879|2.724|3.654, which is similar with prev hard.
# 0.001|0.3|0|0.57
# 4.456|5.879|1.089|3.602
RELATION=atomic18 MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    THRESHOLD_TYPE=proportion THRESHOLD="0.005|0.3|0.1|0.6" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/proportion_thresh/atomic_prop_0.333_thresh_0.005_0.3_0.1_0.6.csv \
    python ./data_preparation/filter_and_select.py

RELATION=atomic18 MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    THRESHOLD_TYPE=hard THRESHOLD="4.36|5.8|4.0|3.6|4.0" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/atomic_pseudo_triple_maxproportion_0.333_new.csv \
    python ./data_preparation/filter_and_select.py

RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    THRESHOLD_TYPE=hard THRESHOLD="4.36|5.8|4.0|3.6|4.0" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    python ./data_preparation/filter_and_select.py &

# kgbert test
RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/kgbert_base_rel_special \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    INPUT_NEGATIVE_LOSS=0 \
    THRESHOLD_TYPE=hard  THRESHOLD="4.36|5.8|0.1|0.1|0.1" \
    OUTPUT_FILE=./data/pseudo_label_trn/kgbert_base_rel_special/pseudo_triple_maxproportion_0.333_new.csv \
    python ./data_preparation/filter_and_select.py

