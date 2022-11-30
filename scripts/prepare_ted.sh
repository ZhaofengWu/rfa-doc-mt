#!/bin/bash
# Adapted from https://github.com/idiap/HAN_NMT/blob/2564826d7653eb5c69c982f569d90adb00a15f04/preprocess_TED_zh-en/prepare.sh

set -e

export PYTHONPATH="$(pwd):${PYTHONPATH}"

BPE_TOKENS=30000

if [[ ${#} -ne 3 && ${#} -ne 5 ]]; then
  echo "usage: bash scripts/prepare_ted.sh DATASET_DIR SRC_LANG TGT_LANG (START_STAGE END_STAGE)"
  exit 1
fi

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

max_len=200

dataset_dir=$1
src=$2
tgt=$3
pair=$src-$tgt
output_dir=data/ted/${pair}
intermediate_dir=${output_dir}/intermediate

start_stage=1
end_stage=9999
if [[ ${#} -eq 5 ]]; then
    start_stage=$4
    end_stage=$5
fi

mkdir -p ${output_dir} ${intermediate_dir}

if [[ ${start_stage} -le 1 && ${end_stage} -ge 1 ]]; then
  echo "Parsing data"
  pair_dir=${dataset_dir}/texts/${src}/${tgt}/${pair}
  if [ ! -d ${pair_dir} ]; then
    tar xzf ${pair_dir}.tgz -C ${dataset_dir}/texts/${src}/${tgt}
  fi
  rm ${intermediate_dir}/* || : # because the Python script append to certain files, we need to make sure they start empty
  time python scripts/parse_ted.py ${pair_dir} ${intermediate_dir} ${src} ${tgt}
fi

if [[ ${start_stage} -le 2 && ${end_stage} -ge 2 ]]; then
  for lang in ${src} ${tgt}; do
    echo "Tokenizing ${lang}"
    time cat ${intermediate_dir}/train.raw.$lang | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $moses_scripts/tokenizer/tokenizer.perl -a -l $lang \
    > ${intermediate_dir}/train.tok.$lang
  done
fi

if [[ ${start_stage} -le 3 && ${end_stage} -ge 3 ]]; then
  for lang in ${src} ${tgt}; do
    echo "Training true caser"
    time $moses_scripts/recaser/train-truecaser.perl -model ${intermediate_dir}/truecase-model.$lang -corpus ${intermediate_dir}/train.tok.$lang
    echo "Applying true caser"
    time $moses_scripts/recaser/truecase.perl < ${intermediate_dir}/train.tok.$lang > ${intermediate_dir}/train.tc.$lang -model ${intermediate_dir}/truecase-model.$lang
  done
fi

# valid/test sets
if [[ ${start_stage} -le 4 && ${end_stage} -ge 4 ]]; then
  for split in valid test; do
    for lang in ${src} ${tgt}; do
      echo "Processing ${split} ${lang}"
      time $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < ${intermediate_dir}/$split.raw.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model ${intermediate_dir}/truecase-model.$lang \
      > ${intermediate_dir}/$split.tc.$lang
    done
  done
fi

bpe_corpus=${intermediate_dir}/bpe_corpus.txt
bpe_code=${output_dir}/bpe_code.txt

if [[ ${start_stage} -le 5 && ${end_stage} -ge 5 ]]; then
    echo "Learning bpe"

    rm ${bpe_corpus} || :
    for lang in ${src} ${tgt}; do
        cat ${intermediate_dir}/train.tc.${lang} >> ${bpe_corpus}
    done

    time python ${BPEROOT}/learn_bpe.py -s ${BPE_TOKENS} < ${bpe_corpus} > ${bpe_code}
fi

if [[ ${start_stage} -le 6 && ${end_stage} -ge 6 ]]; then
    for split in train valid test; do
        for lang in ${src} ${tgt}; do
            echo "Applying bpe on ${split} ${lang}"
            time python $BPEROOT/apply_bpe.py -c ${bpe_code} < ${intermediate_dir}/${split}.tc.${lang} > ${output_dir}/${split}.${lang}
        done
    done
fi

if [[ ${start_stage} -le 7 && ${end_stage} -ge 7 ]]; then
  echo "Cleaning up $(du -sh ${intermediate_dir}) intermediate data"
  rm -r ${intermediate_dir}
fi

if [[ ${start_stage} -le 8 && ${end_stage} -ge 8 ]]; then
    rm -r ${output_dir}/binary || :
    time fairseq-preprocess --user-dir $(pwd) --task doc_translation \
        --source-lang ${src} --target-lang ${tgt} \
        --trainpref ${output_dir}/train --validpref ${output_dir}/valid --testpref ${output_dir}/test \
        --destdir ${output_dir}/binary
    cp ${intermediate_dir}/*.doc_end_indices ${intermediate_dir}/*.metadata.{$src,$tgt} ${output_dir}
    cp ${intermediate_dir}/*.doc_end_indices ${intermediate_dir}/*.metadata.{$src,$tgt} ${output_dir}/binary
fi
