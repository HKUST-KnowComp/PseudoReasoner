# check proportion.
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/proportion_thresh/prop_0.333_thresh_0.005_0.3_0.1_0.6.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo \
    --experiment_name ckbp_gpt2xl_score_gpt2xl_filter_prop_0.333_thresh_0.005_0.3_0.1_0.6 --pretrain_pseudo_steps 3000

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/proportion_thresh/prop_0.333_thresh_0.001_0.3_0_0.57.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo \
    --experiment_name ckbp_gpt2xl_score_gpt2xl_filter_prop_0.333_thresh_0.001_0.3_0_0.57 --pretrain_pseudo_steps 2000
# atomic test

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/atomic_pseudo_triple_maxproportion_0.333_new.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo \
    --experiment_name ckbp_gpt2xl_score_gpt2xl_filter_atomic_new --pretrain_pseudo_steps 3000

# test bert: data/pseudo_label_trn/kgbert_base_rel_special/pseudo_triple_maxproportion_0.333_new.csv
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/kgbert_base_rel_special/pseudo_triple_maxproportion_0.333_new.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _from_best_pseudo_kgbert_label --pretrain_pseudo_steps 3000


# pseudo_triple_maxproportion_0.333_new.csv
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/pseudo_triple_maxproportion_0.333_new.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name _from_best_pseudo_new --pretrain_pseudo_steps 3000


# with special token
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 32 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/pseudo_triple_maxproportion_0.333.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name from_best_pseudo --pretrain_pseudo_steps 100000

