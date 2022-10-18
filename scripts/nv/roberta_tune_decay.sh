# pseudo eval every 125, pseudo lr 1e-5, pseudo lr decay 0.9, decay step [50, 100, 125, 200, 250, 500]
SEED=$1 # [100, 101, 102]
PSEUDO_LRDECAY=$2 # [0.9, 0.95, 0.98]
PSEUDO_DECAY_EVERY=$3 # [50, 100, 125, 200, 250, 500]

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm /mount/checkpoint/roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir /results \
    --train_csv_path /mount/data/emnlp2021/train.csv \
    --evaluation_file_path /mount/data/evaluation_set.csv \
    --relation_as_special_token \
    --seed $SEED --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path /mount/pseudo_trn/prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "" \
    --show_result_only --pseudo_eval_every 125 \
    --pretrain_pseudo_steps 10000 --steps 3000 --pseudo_lr 1e-5 --pseudo_lrdecay $PSEUDO_LRDECAY --pseudo_decay_every $PSEUDO_DECAY_EVERY
