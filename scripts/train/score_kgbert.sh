CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/scoring_kgbert.py \
    --ptlm bert-base-uncased \
    --model_path /mount/checkpoint/best_model_seed_100.pth \
    --tokenizer /mount/checkpoint/best_tokenizer/ \
    --output_dir /results \
    --scorer_name bert_base_eval32_rel_special \
    --evaluation_file_path /mount/candidates/test.csv
