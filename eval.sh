#!/bin/bash

if [[ ${#} -eq 3 ]]; then
    echo ${1} ${2} ${3}
elif [[ ${#} -eq 4 ]]; then
    echo $1 $2 $3 $4
else
    echo "wrong number of arguments"
    exit 1;
fi

DATASET_DIR=${1}
LANG=${2}
model_path=${3}
EXTRA_FLAGS=""
if [[ ${#} -ge 4 ]]; then
    EXTRA_FLAGS=$4
fi
SPLIT="test"

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

export PYTHONPATH="$(pwd):${PYTHONPATH}"

fairseq-generate ${DATASET_DIR}/binary \
--gen-subset ${SPLIT} \
--user-dir $(pwd) \
--task doc_translation \
--path $model_path \
--left-pad-source False \
--fp16 \
--fp16-init-scale 32 \
--fp16-scale-window 1024 \
--beam 4 --max-len-a ${max_len_a} --max-len-b ${max_len_b} --batch-size 1 --remove-bpe ${EXTRA_FLAGS}

