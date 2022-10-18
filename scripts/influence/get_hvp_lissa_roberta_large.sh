CUDA_VISIBLE_DEVICES=1 python data_preparation/get_hvp_lissa.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --train_file_path "./data/ckbp_csv/emnlp2021/train.csv" \
    --eval_file_path "./data/evaluation_set.csv" \
    --num_train 32000 \
    --damping 0.01 \
    --c 1e7 \
    --epoch 10 --seed 102
## same command for get_hvp_lissa.py
# if Out-of-Memory for 12GB GPU, set train_batch_size to 8.
