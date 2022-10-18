CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/scoring_kgbert.py \
    --ptlm roberta-large \
    --model_path "results/noisy_student/test/roberta-large_bs64_evalstep250/pseudo_lr1e-05_decay1_step500_evalstep250_dropout_0.5/iter_4/best_model_seed_100.pth" \
    --tokenizer "results/noisy_student/test/roberta-large_bs64_evalstep250/pseudo_lr1e-05_decay1_step500_evalstep250_dropout_0.5/iter_4/best_tokenizer" \
    --output_dir "results/noisy_student/test/roberta-large_bs64_evalstep250/pseudo_lr1e-05_decay1_step500_evalstep250_dropout_0.5" \
    --scorer_name "iter_4" \
    --evaluation_file_path "data/pseudo_label_trn/roberta_large/hard_4.36_5.8_0.1_0.1_0.1_new.csv"
