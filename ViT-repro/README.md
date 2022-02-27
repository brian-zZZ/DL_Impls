## Dataset
Flower classification dataset

## Only fine-tuning code. Excute the ```train.py``` with scripts below to fine-tune.
* Run with single GPU. Specify with ```device``` setting.
* Change **```from vit_model import vit_base_patch16_224_in21k as create_model```** to switch model.
```
python train.py \
    --num_classes 5 \
    --epochs 10 --batch_size 8 --lr 0.001 \
    --data_path '../datasets/flower_data/ \
    --weights './weights/vit_base_patch16_224_in21k.pth' \
    --device 'cuda:0'
```
