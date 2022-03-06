# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
A Pytorch implementation. **Pre-train BERT from scratch.**

## Pre-training
### 0) Corpus preparation
BERT is originally pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words), for simplicity,
here we pre-train on a relatively small corpus: 
[wikitext-2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip').
> Note that wikitext-2 corpus is borrowed from chapter 
['The Dataset for Pretraining BERT'](https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html)
of Dive Into Deep Learing (Li Mu, etc).
- Download and extract the wikitext-2 corpus to ```'./data/wikitext-2/'```.

### 1) Build vocab
```bash
python vocab_builder.py -c ./data/wikitext-2/wiki.train.tokens \
  -o ./data/wikitext-2/wikitext-vocab.pkl -m 5
```

### 2) Pre-train BERT from scratch
```bash
CUDA_VISIBLE_DEVICES=6,7 python main_pretrain.py \
  -c ./data/wikitext-2/wiki.train.tokens \
  -v ./data/wikitext-2/wikitext-vocab.pkl \
  -o ./output/ \
  -b 512 --with_cuda --log_freq 20
```

## Fine-tuning
This part is yet to be updated.
- Load pre-trained model, attach linear projection layers with respect to specific downstream task, then fine-tune.

## Ackownledgement
* This implementation is heavily borrowed from [BERT-torch](https://github.com/codertimo/BERT-pytorch), Junseong Kim, Scatter Lab.
  Thanks a lot.
