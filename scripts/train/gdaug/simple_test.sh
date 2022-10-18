CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/gdaug/test \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path results/comet/csv/gpt2_num1_sample.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "" \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 

# bert-base
CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo/gdaug/test \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path results/comet/csv/gpt2_num1_sample.csv \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name "" \
    --pretrain_from_path results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 