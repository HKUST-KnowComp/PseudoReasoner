# no filter
CUDA_VISIBLE_DEVICES=2 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 6000 --steps 3000 


# diversity

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_div.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "div" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 3000 --steps 3000 

# kg-bert

CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 100 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_kgbert_0.5_0.9.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "kgbert_" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 500 --steps 3000 

# influence

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_influence_only.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "influence_only" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 6000 --steps 3000 

# influence + kgbert

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_kgbert_0.5_0.9_influence_seed100.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "influence_kgbert_seed100" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 6000 --steps 3000 

# kgbert + div
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_div_kgbert.csv" \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "div_kgbert" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 6000 --steps 3000 

# combo

CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir "results/pseudo/gdaug/gpt2_small" \
    --train_csv_path "data/ckbp_csv/emnlp2021/train.csv" \
    --relation_as_special_token \
    --seed 102 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path "results/comet/csv/gpt2_unlabeled/comet_gpt2_unlabeled_kgbert_0.5_0.9_influence_seed100_div_20000_label_0.csv" \
    --pretrain_pseudo_epochs 2 --resume_train_from_best_pseudo --experiment_name "combo" \
    --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --pretrain_pseudo_steps 6000 --steps 3000 
