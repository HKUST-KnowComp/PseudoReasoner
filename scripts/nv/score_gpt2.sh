CUDA_VISIBLE_DEVICES=0 TOKENIZER=/mount/checkpoint/best_tokenizer \
    GPT2_MODEL=/mount/checkpoint/best_model_seed_100 \
    CANDIDATE_SCORING_FILE=/mount/candidates/xWant.csv \
    VALID_BATCH_SIZE=1024 \
    SCORER_NAME=gpt2_bs32 OUT_DIR=/results python models/gpt2/scoring_gpt2.py

ngc batch run --name "Scoring xWant using ckbp_gpt2" --preempt RUNONCE --min-timeslice 0s --total-runtime 0s --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "cd /mount/workspace/PseudoReasoner; CUDA_VISIBLE_DEVICES=0 TOKENIZER=/mount/checkpoint/best_tokenizer \ GPT2_MODEL=/mount/checkpoint/best_model_seed_100 \ CANDIDATE_SCORING_FILE=/mount/candidates/xWant.csv \ VALID_BATCH_SIZE=1024 \ SCORER_NAME=gpt2_bs32 OUT_DIR=/results python models/gpt2/scoring_gpt2.py " --result /results --image "nvidian/sae/ckbp_torch:torch1.8-cu111-hf4.15-pyg" --org nvidian --team sae --datasetid 98069:/mount/checkpoint --datasetid 97992:/mount/candidates --workspace jPxU2ic0Td6ZAPmK5L9Iew:/mount/workspace:RW