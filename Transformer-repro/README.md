# Attention is all you need: A Pytorch Implementation

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)". 


A novel sequence to sequence framework utilizes the **self-attention mechanism**, instead of Convolution operation or Recurrent structure, and achieve the state-of-the-art performance on **WMT 2014 English-to-German translation task**. (2017/06/12)


<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


# Usage

## WMT'16 Multimodal Translation: de-en
Before running the scripts. Install env dependency.
```pip install -r 'requirements.txt'```

### 0) Download the spacy language model.
- Way 1
```bash
python -m spacy download en
python -m spacy download de
```
- Way 2. If Way 1 encounter connection error.
  - Download [en_core_web_sm-2.3.0](https://objects.githubusercontent.com/github-production-release-asset-2e65be/84940268/cb2afe00-ab05-11ea-8bb0-700d075263a6?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220227T144450Z&X-Amz-Expires=300&X-Amz-Signature=6cd773612329fbc6778823277174151ce17efc10b18c04269ac4dbedec0f24e3&X-Amz-SignedHeaders=host&actor_id=49059409&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_core_web_sm-2.3.0.tar.gz&response-content-type=application%2Foctet-stream) and [de_core_news_sm-2.3.0](https://objects.githubusercontent.com/github-production-release-asset-2e65be/84940268/4fc94c80-ab05-11ea-975e-c073a9557566?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220227T145721Z&X-Amz-Expires=300&X-Amz-Signature=f110df827abfb6c099323f0d0da63a22caeadcc1b5bd062dc574cd2295a22512&X-Amz-SignedHeaders=host&actor_id=49059409&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Dde_core_news_sm-2.3.0.tar.gz&response-content-type=application%2Foctet-stream) to local.
  - Pip install these two packages identically
  ```pip install 'some_dir/pk'```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
Single GPU. Specify the CUDA device number by ```--device_num```.
```bash
python train.py --data_pkl m30k_deen_shr.pkl \
  -b 128 -warmup 128000 --epoch 400 \
  --embs_share_weight \
  --proj_share_weight \
  --output_dir output \
  --device_num 0 \
  --label_smoothing
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

## [(WIP)] WMT'17 Multimodal Translation: de-en w/ BPE 
### 1) Download and preprocess the data with bpe:

> Since the interfaces is not unified, you need to switch the main function call from `main_wo_bpe` to `main`.

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) Train the model
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) Test the model (not ready)
- TODO:
	- Load vocabulary.
	- Perform decoding after the translation.
---
# Performance
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
- The project structure, some scripts and the dataset preprocessing steps are heavily borrowed from [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
- Thanks for the suggestions from @srush, @iamalbert, @Zessay, @JulesGM, @ZiJianZhao, and @huanghoujing.