# kg_bert_base filter, 0.5-0.9:
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/iter_adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/bert_base_filter/0.5_0.9.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name 0.5_0.9 \
    --pretrain_from_path results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 

# no filter.
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/iter_adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/w_kgbert_fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name no_filter \
    --pretrain_from_path results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 1000 --steps 1000 

# influence_only

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --pretrain_pseudo_epochs 2 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/iter_adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/bert_base_filter/influence_only.csv \
    --resume_train_from_best_pseudo --experiment_name influence_only \
    --pretrain_from_path results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 

# influence + kgbert

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/iter_adversarial \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/bert_base_filter/influence_0.5_0.9.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name influence_kgbert_filter \
    --pretrain_from_path results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 