CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name fixed \
    --pretrain_pseudo_steps 6000 --steps 3000 

# from scratch
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 5e-6 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/adversarial/fixed_0.5_0.9.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name fixed_0.5_0.9_roberta \
    --pretrain_pseudo_steps 6000 --steps 3000 
# from scratch
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/adversarial/bert_base/0.5_0.9.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name 0.5_0.9 \
    --pretrain_pseudo_steps 6000 --steps 3000 

