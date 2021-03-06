# gMLP: [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050), Google Brain, June 2021.
gMLP performs better than MLP-Mixer, further more, gMLP attains competitive results in various tasks, including both CV and NLP. \
The most impressive conclusion from the paper is that Attention may be unnecessary for Transformers, 
meanwhile Attention may introduce more inductive bias, which is importance for downstream tasks that require cross-sentence infomation.

**Attention: This implementation can load the official pre-trained weights, both share the same namespace.**
> Note: The official pre-trained weights refer to the [```timm``` library](https://github.com/rwightman/pytorch-image-models) release.

## Usage
### Dataset
[Flower classification dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)

### Pre-train
In the original paper, the models are pre-trained in ImageNet-1k , which are both natural image datasets.

### Fine-tune
Excute the ```main_finetune.py``` with scripts below to fine-tune with pre-trained weights.
* Run on single GPU. Specify with ```device``` setting.
* Change **```from gmlp import gmlp_s16_224 as gmlp```** to switch model.
```bash
python main_finetune.py --num_classes 5 \
    --epochs 10 --batch-size 16 --lr 1e-4 \
    --data-path '/home/brian/datasets/flower_data/flower_photos' \
    --pretrained --device 'cuda:0'
```

* Results

> From top-left to bottom-right: train acc, loss and val acc, loss
<p align="center">
<img src="result.png" width="800">
</p>

From the figure, we can tell that 10 epochs are too much for such a small dataset to fine-tune, resulting in a over-fitting phenomenon.

## Reference
- [```timm``` library](https://github.com/rwightman/pytorch-image-models), Ross Wightman.
- [mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch), lucidrains.
- [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch), xiaoma.