CUDA_VISIBLE_DEVICES=0 python data_preparation/get_hvp.py \
    --ptlm bert-base-uncased \
    --checkpoint_path "./results/bert-base-uncased_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/bert-base-uncased_bs64_evalstep250/best_tokenizer" \
    --train_file_path "./data/ckbp_csv/emnlp2021/train.csv" \
    --eval_file_path "./data/evaluation_set.csv" \
    --num_train 32000 \
    --weight_decay 0.01 \
    --damping 0.01 \
    --gradient_accumulation_steps 10 \
    --c 1e7 \
    --epoch 10