## using backtranslation
CUDA_VISIBLE_DEVICES=0 python baseline/uda/backtrans_en_fr.py \
    --unsup_data_dir= path/to/unsup_data \
    --batch_size=32 

for seed in 100 101 102
do
  for unsup_ratio in 1
  do
    for uda_coeff in 1
    do
      CUDA_VISIBLE_DEVICES=0 python baseline/uda/train.py \
      --ptlm roberta-large \
      --lr 1e-5 \
      --steps 3000 \
      --batch_size 64 \
      --test_batch_size 128 \
      --output_dir results \
      --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
      --pretrain_from_path "results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
      --relation_as_special_token \
      --seed $seed \
      --unsup_data_dir path/to/unsup_data \
      --aug_ops bt_en2fr \
      --unsup_ratio $unsup_ratio \
      --uda_coeff $uda_coeff \
      --experiment_name _resume_uda_backtrans_ur${unsup_ratio}_uc${uda_coeff}
    done 
  done
done