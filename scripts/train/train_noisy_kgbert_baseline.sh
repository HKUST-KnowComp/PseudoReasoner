# noisy training
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_kgbert_baseline.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results/noisy \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --noisy_training --p_noisy 0.5 \
    --save_best_model \
    --seed 100 --batch_size 64 --test_batch_size 128 --save_best_model --experiment_name ""