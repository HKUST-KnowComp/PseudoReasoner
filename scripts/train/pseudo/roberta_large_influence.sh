# no filter.
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/influence_test \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/w_kgbert_fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name no_filter \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 3000 --steps 3000 

# influence_only

CUDA_VISIBLE_DEVICES=3 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --pretrain_pseudo_epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/influence_1e8 \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/roberta_large_filter/influence_only_1e8_seed101.csv \
    --resume_train_from_best_pseudo --experiment_name influence_only_seed101 \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 3000 --steps 3000 

# influence + kgbert

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/influence_1e8 \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/roberta_large_filter/influence_1e8_seed100_0.5_0.9.csv \
    --pretrain_pseudo_epochs 3 --resume_train_from_best_pseudo --experiment_name seed100 \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 
