#!/bin/bash

if [[ ${#} -eq 3 ]]; then
    echo ${1} ${2} ${3}
elif [[ ${#} -eq 4 ]]; then
    echo $1 $2 $3 $4
else
    echo "wrong number of arguments"
    exit 1;
fi

DATASET=${1}
LANG=${2}
model_path=${3}
EXTRA_FLAGS=""
if [[ ${#} -eq 4 ]]; then
    EXTRA_FLAGS=$4
fi
SPLIT="test_consistency"

echo $MODEL_PATH

export MKL_SERVICE_FORCE_INTEL=1
export PYTHONPATH="$(pwd):${PYTHONPATH}"

fairseq-validate ${DATASET}/binary \
--valid-subset ${SPLIT} \
--user-dir $(pwd) \
--task doc_translation \
--path $model_path \
--left-pad-source False \
--fp16 \
--fp16-init-scale 32 \
--fp16-scale-window 1024 \
--max-tokens 3072 ${EXTRA_FLAGS}
