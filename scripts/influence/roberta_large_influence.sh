
CUDA_VISIBLE_DEVICES=0 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_500000000.0_epoch_10/seed_101.pth" \
    --candidate_file_path "data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv" \
    --start 0 --end 250000
CUDA_VISIBLE_DEVICES=1 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_500000000.0_epoch_10/seed_101.pth" \
    --candidate_file_path "data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv" \
    --start 250000 --end 500000
CUDA_VISIBLE_DEVICES=2 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_500000000.0_epoch_10/seed_101.pth" \
    --candidate_file_path "data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv" \
    --start 500000 --end 750000
CUDA_VISIBLE_DEVICES=3 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_500000000.0_epoch_10/seed_101.pth" \
    --candidate_file_path "data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/hard/fixed_prop_0.333_4.36_5.8_4.0_3.6_4.0.csv" \
    --start 750000 --end 950000
