CUDA_VISIBLE_DEVICES=1 TOKENIZER=results/gpt2_bs32_evalstep250_rel_special/best_tokenizer \
    GPT2_MODEL=results/gpt2_bs32_evalstep250_rel_special/best_model_seed_100 \
    CANDIDATE_SCORING_FILE=data/discos_csv/candidates/test.csv \
    SCORER_NAME=gpt2_bs32 python models/gpt2/scoring_gpt2.py