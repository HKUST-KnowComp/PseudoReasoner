# RoBERTa-large
RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/roberta_large \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    INPUT_NEGATIVE_LOSS=0 \
    THRESHOLD_TYPE=hard  THRESHOLD="4.36|5.8|0.1|0.1|0.1" \
    OUTPUT_FILE=./data/pseudo_label_trn/roberta_large/hard_4.36_5.8_0.1_0.1_0.1.csv \
    python ./data_preparation/filter_and_select.py

# RoBERTa-large
RELATION=all MAX_PROPORTION=0.333 \
    PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/roberta_large \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    CANDIDATE_PATH=data/discos_csv/candidates_trunc \
    INPUT_NEGATIVE_LOSS=0 \
    THRESHOLD_TYPE=hard  THRESHOLD="4.36|5.8|0.1|0.1|0.2" \
    OUTPUT_FILE=./data/pseudo_label_trn/roberta_large/hard_4.36_5.8_0.1_0.1_0.2.csv\
    python ./data_preparation/filter_and_select.py

