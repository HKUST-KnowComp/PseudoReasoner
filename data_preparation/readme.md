# Get teacher model scores for unlabeled data

Score using GPT2. 

```
CUDA_VISIBLE_DEVICES=1 TOKENIZER=results/gpt2_bs32_evalstep250_rel_special/best_tokenizer \
    GPT2_MODEL=results/gpt2_bs32_evalstep250_rel_special/best_model_seed_100 \
    CANDIDATE_SCORING_FILE=xWant.csv \
    SCORER_NAME=gpt2_bs32 python models/gpt2/scoring_gpt2.py
```


# Filter pseudo labels based on thresholding

```
PSEUDO_LABEL_SCORE_DIR=data/pseudo_label_scores/ckbp_gpt2xl_eval10 \
    FLUENCY_LOSS_DIR=data/pseudo_label_scores/gpt2xl_zeroshot \
    INPUT_NEGATIVE_LOSS=1 THRESHOLD="4.36|5.8|4.0|3.7|2.8|2.0" \
    OUTPUT_FILE=./data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/pseudo_examples.csv \
    python ./data_preparation/filter_new.py
```

# Calculate influence score

get HVP

```
CUDA_VISIBLE_DEVICES=1 python data_preparation/get_hvp_lissa.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --train_file_path "./data/ckbp_csv/emnlp2021/train.csv" \
    --eval_file_path "./data/evaluation_set.csv" \
    --num_train 32000 \
    --damping 0.01 \
    --c 1e7 \
    --epoch 10 --seed 102
```

Calc Influence

```
CUDA_VISIBLE_DEVICES=0 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_500000000.0_epoch_10/seed_101.pth" \
    --candidate_file_path path/to/candidate/csv
```