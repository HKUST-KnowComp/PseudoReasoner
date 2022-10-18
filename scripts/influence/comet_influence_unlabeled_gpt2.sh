CUDA_VISIBLE_DEVICES=0 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_100000000.0_epoch_10/seed_100.pth" \
    --candidate_file_path "results/comet/csv/gpt2-xl_unlabeled/comet_gpt2_unlabeled.csv" \
    --start 0 --end 100000
CUDA_VISIBLE_DEVICES=0 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_100000000.0_epoch_10/seed_100.pth" \
    --candidate_file_path "results/comet/csv/gpt2-xl_unlabeled/comet_gpt2_unlabeled.csv" \
    --start 100000 --end 200000
CUDA_VISIBLE_DEVICES=1 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_100000000.0_epoch_10/seed_100.pth" \
    --candidate_file_path "results/comet/csv/gpt2-xl_unlabeled/comet_gpt2_unlabeled.csv" \
    --start 200000 --end 300000
CUDA_VISIBLE_DEVICES=1 python data_preparation/calc_influence_with_hvp.py \
    --ptlm roberta-large \
    --checkpoint_path "./results/roberta-large_bs64_evalstep250/best_model_seed_100.pth" \
    --tokenizer_path "./results/roberta-large_bs64_evalstep250/best_tokenizer" \
    --weight_decay 0.0 \
    --hvp_path "./results/roberta-large_bs64_evalstep250/hvp_lissa_32000_damping_0.01_c_100000000.0_epoch_10/seed_100.pth" \
    --candidate_file_path "results/comet/csv/gpt2-xl_unlabeled/comet_gpt2_unlabeled.csv" \
    --start 300000 --end 400000