#!/bin/bash

if [[ ${#} -eq 6 ]]; then
    echo ${1} ${2} ${3} ${4} ${5} ${6}
elif [[ ${#} -eq 7 ]]; then
    echo $1 $2 $3 $4 $5 $6 $7
else
    echo "wrong number of arguments"
    exit 1;
fi

DATASET=${1}
LANG=${2}
OUTPUT_DIR=${3}
CONTEXT_SIZE=${4}
MAX_TOKENS=${5}
UPDATE_FREQ=${6}
EXTRA_FLAGS=""
if [[ ${#} -eq 7 ]]; then
    EXTRA_FLAGS=$7
fi

export MKL_SERVICE_FORCE_INTEL=1
export PYTHONPATH="$(pwd):${PYTHONPATH}"

path=${OUTPUT_DIR}

lr=1e-3
dim=512
n_heads=8
ffn_dim=1024
wd=1e-4
drop=0.3

valid_subset="valid"  # change to "valid_consistency" when appropriate
best_checkpoint_metric="bleu"  # change to "consistency" when appropriate

max_len_a=x
max_len_b=x
if [[ ${LANG} == "zh-en" ]]; then
    max_len_a=1.2
    max_len_b=70
elif [[ ${LANG} == "en-ru" ]]; then
    max_len_a=1
    max_len_b=50
else
    echo "Unrecognized language pair"
    exit 1;
fi

if [ -d ${OUTPUT_DIR} ]; then
    echo "Output dir exists"
    exit 1;
fi
mkdir -p ${path}

fairseq-train ${DATASET}/binary \
--valid-subset ${valid_subset} \
--user-dir $(pwd) \
--task doc_translation \
--arch doc_translation_transformer_iwslt_de_en \
--left-pad-source False \
--activation-fn gelu \
--share-decoder-input-output-embed \
--optimizer adam \
--clip-norm 0.0 \
--lr ${lr} \
--seed 31415 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt \
--dropout ${drop} \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens ${MAX_TOKENS} \
--max-tokens-valid $(echo "3 * ${MAX_TOKENS} / 2" | bc) \
--update-freq ${UPDATE_FREQ} \
--no-epoch-checkpoints \
--save-dir ${path} \
--encoder-embed-dim ${dim} \
--decoder-embed-dim ${dim} \
--encoder-ffn-embed-dim ${ffn_dim} \
--decoder-ffn-embed-dim ${ffn_dim} \
--encoder-attention-heads ${n_heads} \
--decoder-attention-heads ${n_heads} \
--weight-decay ${wd} \
--warmup-updates 8000 \
--ddp-backend=no_c10d \
--find-unused-parameters \
--fp16 \
--fp16-init-scale 32 \
--fp16-scale-window 1024 \
--eval-bleu \
--eval-bleu-args "{\"beam\": 4, \"max_len_a\": ${max_len_a}, \"max_len_b\": ${max_len_b}}" \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric ${best_checkpoint_metric} \
--maximize-best-checkpoint-metric \
--patience 10 \
--context-size ${CONTEXT_SIZE} \
${EXTRA_FLAGS} 2>&1 | tee ${path}/log.txt

