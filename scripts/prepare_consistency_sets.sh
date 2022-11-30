#!/bin/bash

set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

if [[ ${#} -ne 3 && ${#} -ne 5 ]]; then
  echo "usage: bash scripts/prepare_consistency_sets.sh SRC_LANG TGT_LANG MAIN_DATASET_DIR (START_STAGE END_STAGE)"
  exit 1
fi

if [ ! -d data/consistency_sets ]; then
    git clone git@github.com:lena-voita/good-translation-wrong-in-context.git data/consistency_sets
fi
cd data/consistency_sets
git checkout bb59382f6bc6c01e0cb8e58e370a8dff8198107b
cd ../..

mkdir -p vendor
cd vendor

if [ ! -d mosesdecoder ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi
cd mosesdecoder
git checkout 5cbafabfd5ed2833ca8808bdca6e785935713159
cd ..

if [ ! -d subword-nmt ]; then
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
fi
cd subword-nmt
git checkout 234923ed53c19f17a4456f1316f14dd9e033712b
cd ..

cd ..

moses_scripts=vendor/mosesdecoder/scripts
BPEROOT=vendor/subword-nmt/subword_nmt

src=$1
tgt=$2
main_dataset_dir=$3
main_dataset_intermediate_dir=${main_dataset_dir}/intermediate
input_dir=data/consistency_sets/consistency_testsets/scoring_data
intermediate_dir=${input_dir}/intermediate

start_stage=1
end_stage=9999
if [[ ${#} -eq 5 ]]; then
    start_stage=$4
    end_stage=$5
fi

if [ ! -d ${main_dataset_intermediate_dir} ]; then
    echo "${main_dataset_intermediate_dir} not found"
    exit 1
fi
mkdir -p ${intermediate_dir}

if [[ ${start_stage} -le 1 && ${end_stage} -ge 1 ]]; then
  echo "Parsing data"
  time python scripts/parse_consistency_sets.py ${input_dir} ${intermediate_dir}
fi

if [[ ${start_stage} -le 2 && ${end_stage} -ge 2 ]]; then
  for side in src dst; do
    lang=${src}
    if [[ $side == "dst" ]]; then
        lang=${tgt}
    fi
    echo "Processing ${lang}"
    for file in $(ls ${intermediate_dir}/*.${side}.raw); do
        $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < ${file} | \
        $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
        $moses_scripts/recaser/truecase.perl -model ${main_dataset_intermediate_dir}/truecase-model.$lang \
        > "$(echo $file | rev | cut -d. -f2- | rev).tc"
    done
  done
fi

if [[ ${start_stage} -le 3 && ${end_stage} -ge 3 ]]; then
    for side in src dst; do
        lang=${src}
        if [[ $side == "dst" ]]; then
            lang=${tgt}
        fi
        echo "Applying bpe on ${lang}"
        for file in $(ls ${intermediate_dir}/*.${side}.tc); do
            python $BPEROOT/apply_bpe.py -c ${main_dataset_dir}/bpe_code.txt < ${file} > "$(echo $file | rev | cut -d. -f2- | rev).line"
        done
    done
fi

if [[ ${start_stage} -le 4 && ${end_stage} -ge 4 ]]; then
    for side in src dst; do
        lang=${src}
        if [[ $side == "dst" ]]; then
            lang=${tgt}
        fi
        echo "Merging ${lang}"
        for split in valid test; do
            rm ${intermediate_dir}/${split}.${lang} || :
            if [[ $split == "valid" ]]; then
                files=$(ls ${intermediate_dir}/*_dev.${side}.line)
            else  # ellipsis sets don't mark "_test"
                files=$(ls ${intermediate_dir}/*.${side}.line | grep -v "_dev.")
            fi
            for file in $files; do
                cat ${file} >> ${intermediate_dir}/${split}.${lang}
            done
        done
    done
fi

if [[ ${start_stage} -le 5 && ${end_stage} -ge 5 ]]; then
    rm -r ${main_dataset_dir}/binary || :
    time fairseq-preprocess --user-dir $(pwd) --task doc_translation \
        --source-lang ${src} --target-lang ${tgt} \
        --trainpref ${main_dataset_dir}/train --validpref ${main_dataset_dir}/valid,${intermediate_dir}/valid --testpref ${main_dataset_dir}/test,${intermediate_dir}/test \
        --destdir ${main_dataset_dir}/binary
    cp ${main_dataset_dir}/*.doc_end_indices ${main_dataset_dir}/binary
    cp ${main_dataset_dir}/*.metadata.{$src,$tgt} ${main_dataset_dir}/binary || :  # For OpenSubtitles we don't have these

    for file in $(ls ${main_dataset_dir}/binary/valid1.*); do
        mv ${file} ${main_dataset_dir}/binary/valid_consistency.$(echo $file | rev | cut -d/ -f1 | rev | cut -d. -f2-)
    done
    for file in $(ls ${main_dataset_dir}/binary/test1.*); do
        mv ${file} ${main_dataset_dir}/binary/test_consistency.$(echo $file | rev | cut -d/ -f1 | rev | cut -d. -f2-)
    done
fi
