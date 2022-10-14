# PseudoReasoner: Leveraging Pseudo Labels for Commonsense Knowledge Base Population (Findings of EMNLP 2022)

## Set up

Python env:

```
torch 1.7.1
transformers 4.15.0

```

## Data

Original training data of [CSKB Population](https://github.com/HKUST-KnowComp/CSKB-Population#download-the-data).

Unlabeled data (adapted from ASER) can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EkjTFKfA9gJAvkEDJ49mZmgBkgIl7aKLkV4Wfrg91HeeLg?e=SbsTRr). 

Pseudo labels we produced can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/Et4qyZilZf1OtmaRy_3T1ewBNFiXVRrdOcMBqAXTKnVweA?e=AARZLa). 

## Training

baseline (KG-BERT). RoBERTa-large.

```
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_kgbert_baseline.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --save_best_model \
    --seed 100 --batch_size 64 --test_batch_size 128 --experiment_name ""

```

baseline (GPT2)

```
CUDA_VISIBLE_DEVICES=1 TOKENIZER=gpt2-xl GPT2_MODEL=gpt2-xl TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 VALID_BATCH_SIZE=16 EVAL_EVERY=250 SAVE_BEST=True python models/gpt2/train_gpt2.py
```

PseudoReasoner. RoBERTa-large. Checkpoints can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EqvnhHipZR1Bsg75k4x6EaUBEi3TiN1er1oDUPVC7SwOKg?e=OY5yio).

```
CUDA_VISIBLE_DEVICES=0 python models/pseudo_labeling/train_separate_pseudo.py \
    --ptlm roberta-large \
    --lr 1e-5 \
    --pseudo_lr 1e-5 \
    --epochs 1 \
    --output_dir results/pseudo \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --seed 101 --batch_size 64 --test_batch_size 128 \
    --pseudo_examples_path data/pseudo_label_trn/ckbp_gpt2xl_score_gpt2xl_filter/both_filter.csv \
    --pretrain_pseudo_epochs 3 --resume_train_from_best_pseudo \
    --pretrain_from_path results/roberta-large_bs64_evalstep250/best_model_seed_100.pth \
    --pretrain_pseudo_steps 6000 --steps 3000 
```