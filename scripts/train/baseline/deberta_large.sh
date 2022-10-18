CUDA_VISIBLE_DEVICES=1 python models/pseudo_labeling/train_kgbert_baseline.py \
    --ptlm microsoft/deberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --save_best_model \
    --seed 102 --batch_size 64 --test_batch_size 128 --eval_every 250 --save_best_model --experiment_name "" --steps 6000

CUDA_VISIBLE_DEVICES=2 python models/pseudo_labeling/train_kgbert_baseline.py \
    --ptlm microsoft/deberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --save_best_model \
    --seed 101 --batch_size 64 --test_batch_size 128 --eval_every 125 --save_best_model --experiment_name ""