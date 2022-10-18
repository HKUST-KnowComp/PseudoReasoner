CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_7.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_7.0 \
    --pretrain_pseudo_steps 3000 --steps 3000
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_6.5.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_6.5 \
    --pretrain_pseudo_steps 3000 --steps 3000
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_6.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_6.0 \
    --pretrain_pseudo_steps 3000 --steps 3000
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_5.5.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_5.5 \
    --pretrain_pseudo_steps 3000 --steps 3000
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_5.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_5.0 \
    --pretrain_pseudo_steps 3000 --steps 3000

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_4.5.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_4.5 \
    --pretrain_pseudo_steps 3000 --steps 3000
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_4.0 \
    --pretrain_pseudo_steps 3000 --steps 3000

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/prop_0.333_4.36_5.8_4.0_3.6_3.5.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _4.36_5.8_4.0_3.6_3.5 \
    --pretrain_pseudo_steps 3000 --steps 3000