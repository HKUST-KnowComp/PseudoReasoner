CUDA_VISIBLE_DEVICES=0 TOKENIZER=gpt2 \
GPT2_MODEL=gpt2 \
TRAIN_DATA_PATH=data/ckbp_csv/emnlp2021/train.csv \
TRAIN_BATCH_SIZE=64 \
SAVE_LAST_CHECKPOINT=True TRAIN_EPOCHS=3 python models/gpt2/train_comet_gpt2.py
