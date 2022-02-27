## Dataset
Flower classification dataset

## Only fine-tuning code. Excute the ```train.py``` with scripts below to fine-tune.
* Run with single GPU.
* Change ```from vit_model import **vit_base_patch16_224_in21k** as create_model``` to switch model
```
python train.py --num_classes 5\
    --epochs 10 --lr 0.001 \
    --data_path '../datasets/flower_data/ \
    --model-name '
```
