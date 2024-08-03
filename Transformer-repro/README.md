# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)". 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)


<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


# Usage
Before running the scripts. Install env dependency. \
```pip install -r 'requirements.txt'```

## WMT'16 Multimodal Translation: de-en
### 0) Download the spacy language model.
- Way 1
```bash
python -m spacy download en
python -m spacy download de
```
- Way 2. If Way 1 encounter connection error.
  - Download [en_core_web_sm-2.3.0](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz) and [de_core_news_sm-2.3.0](https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.3.0/de_core_news_sm-2.3.0.tar.gz) to local.
  - Pip install these two packages identically. \
  ```pip install 'some_dir/pk'```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab \
  -save_data ./data/multi30k/m30k_deen_shr.pkl
```

### 2) Train the model
Single GPU. Specify the CUDA device number by ```--device_num```.
```bash
python train.py --data_pkl ./data/multi30k/m30k_deen_shr.pkl \
  --output_dir ./output/NMT_Task1 \
  -b 256 -warmup 128000 --epoch 400 \
  --embs_share_weight \
  --proj_share_weight \
  --device_num 0 \
  --use_tb \
  --label_smoothing
```

### 3) Test the model
```bash
python translate.py -data_pkl ./data/multi30k/m30k_deen_shr.pkl \
  -model ./output/NMT_Task1/model.chkpt \
  -output ./output/NMT_Task1/prediction.txt
```

## WMT'17 Multimodal Translation: de-en w/ BPE
### 1) Preprocess the data with bpe
#### Option1: Locally
***The advantage of operating locally is that this way can be easily costimize to other local dataset.***
- Download [wmt17 training set](http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz) and
[wmt17 dev set](http://data.statmt.org/wmt17/translation-task/dev.tgz)
- ***Specify ```*SOURCE``` locally and correctly before execution***
```bash
python preprocess_local.py --raw_dir ./data/raw_deen --bpe_dir ./data/bpe_deen \
  --save_data bpe_vocab.pkl --codes codes.txt --prefix deen --verbose
```

#### Option2: Online
> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main` at ```preprocess.py```.
```bash
python preprocess.py -raw_dir ./data/raw_deen -data_dir ./data/bpe_deen \
  -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py --data_pkl ./data/bpe_deen/bpe_vocab.pkl \
  --train_path ./data/bpe_deen/deen-train \
  --val_path ./data/bpe_deen/deen-val \
  --output_dir ./output/NMT_Task2 \
  --epoch 400 -b 128 -warmup 128000 \
  --embs_share_weight --proj_share_weight \
  --use_tb --device_num 0 --label_smoothing
```

### 3) Test the model (Supplement done)
```bash
python translate.py -data_pkl ./data/bpe_deen/bpe_vocab.pkl \
  -test_path ./data/bpe_deen/deen-test \
  -model ./output/NMT_Task2/model.chkpt \
  -output ./output/NMT_Task2/prediction.txt
```

---
# Performance (task1)
## Training

<p align="center">
<img src="https://i.imgur.com/S2EVtJx.png" width="400">
<img src="https://i.imgur.com/IZQmUKO.png" width="400">
</p>

- Parameter settings:
  - batch size 256 
  - warmup step 4000 
  - epoch 200 
  - lr_mul 0.5
  - label smoothing 
  - do not apply BPE and shared vocabulary
  - target embedding / pre-softmax linear layer weight sharing. 
 
  
## Testing 
- coming soon.
---
# TODO
  - Evaluation on the generated text.
  - Attention weight plot.
---
# Acknowledgement
- The byte pair encoding parts are borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt/).
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from 
  [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- This repo is heavily borrowed from [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
  Thanks a lot.
