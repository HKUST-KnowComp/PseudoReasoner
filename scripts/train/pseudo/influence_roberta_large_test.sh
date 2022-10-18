CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 5e-6 \
    --epochs 1 \
    --output_dir results/pseudo/ckbp_gpt2xl_score_gpt2xl_filter/influence_test \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path /path/to/pseudo_labeled_data \
    --pretrain_pseudo_epochs 1 --resume_train_from_best_pseudo --experiment_name name \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 

