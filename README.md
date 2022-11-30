# Modeling Context With Linear Attention for Scalable Document-Level Translation

The official implementation for our paper (https://arxiv.org/abs/2210.08431):

```bibtex
@inproceedings{wu-etal-2022-modeling,
    title = "Modeling Context With Linear Attention for Scalable Document-Level Translation",
    author = "Zhaofeng Wu and Hao Peng and Nikolaos Pappas and Noah A. Smith",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

# Environment Setup

We performed all experiment in a conda environment with Python 3.7.9.

```bash
pip install -r requirements.txt  # depending on your hardware setup, you may need to install pytorch separatly; see the instruction on the official website
python setup.py install  # you need to adjust the compute_* and sm_* to match your GPU version; see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
```

We noticed that some machines may run into some issues with our original environment. Trying using `numpy==1.20.0` if you do.

# Data

We downloaded the IWSLT (i.e. TED) data from https://wit3.fbk.eu/2015-01 and OpenSubtitles data according to https://github.com/lena-voita/good-translation-wrong-in-context#cadec-data (the context aware version).

You can find scripts that processe these raw data into a form that fairseq can consume in `scripts/*.sh`. `prepare_ted.sh` is for IWSLT, `prepare_processed_opensubtitles.sh` is for OpenSubtitles, and `prepare_consistency_sets.sh` adds the consistency data in https://github.com/lena-voita/good-translation-wrong-in-context into the OpenSubtitles data.

# Pretrained Models

You can find our pretrained models at https://huggingface.co/ZhaofengWu/rfa-doc-mt-models

# Evaluation

Use something like the following, but first change `'random_matrices_path'` to point to the `random_matrices` directory in the cloned repo (absolute path). You don't need this flag for baseline transformer models. Notice that `c`, for `context`, is the window length minus one.
```bash
bash eval.sh data/ted/zh-en zh-en models/iwslt/rfa_c3_sgate.ckpt "--use-sep --model-overrides {'random_matrices_path':'/absolute/path/to/rfa-doc-mt/random_matrices','context_size':3,'right_context_size':0} --doc-bleu --scoring sacrebleu"
```
When evaluating an OpenSubtitles model (for BLEU), because of its special data format where all documents consist of 4 sentences, add `--seq2seq` at the end before the final `"`, and also `--context-size 3` for the c3 (i.e. L=4) model. For the speed benchmark, add `,'cuda_inference':True` before the closing `}` (only for RFA models) and `--add_timer` after. You should get the same BLEU numbers as Table 1 in our paper, though depending on your hardware/software environment, whether you use cuda inference, and whether you directly score using `fairseq-generate` or `fairseq-score` after `fairseq-generate`'s output, you may see variation within around 0.1 BLEU.

For evaluating consistency, use the following flags
```bash
bash eval_consistency.sh data/opensubtitles/en-ru en-ru ckpts/opensubtitles_rfa_l4_sgate.ckpt "--use-sep --model-overrides {'random_matrices_path':'/absolute/path/to/RFA-fairseq/random_matrices','context_size':3,'right_context_size':0} --seq2seq"
```

In our synthetic speed benchmark, we simulate a larger-than-one batch size with the `--duplicate-input-times` when decoding. Please refer to our paper for more details.

# Training Your Own Model

Again because of the difference in format between IWSLT and OpenSubtitles, the flags you need to set are slightly different. You can load the corresponding model with `fairseq-generate` and inspect the original args (note that the first args that are by default printed are the inference args, not the orignal training args; you need to print them yourself). For example, train the IWSLT baseline and the gated RFA model with, respectively:
```bash
bash train_baseline.sh data/ted/zh-en zh-en output_dir 3 4096 4 "--use-sep --full-supervision --extra-padding"
bash train_rfa.sh data/ted/zh-en zh-en output_dir 3 4096 4 256 32 "--use-sep --full-supervision --extra-padding --decay-sentential-gate --decay-gate-bias 2"
```

We slightly cleaned/reorganized our codebase before releasing it. We don't expect it to break anything, but if you see anything unexpected, feel free to open an issue!
